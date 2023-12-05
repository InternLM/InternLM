import multiprocessing as mp
import os
import random
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
from tqdm import tqdm

import internlm
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config
from internlm.core.scheduler import SchedulerMetricHook
from internlm.initialize.launch import args_sanity_check
from internlm.model.loss import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex
from internlm.train import (
    get_train_data_loader,
    get_validation_data_loader,
    initialize_model,
    initialize_optimizer,
)
from internlm.utils.evaluation import switch_evaluation_no_pipeline_scheduler
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

TOTAL_STEPS = 300
config = Config(
    dict(
        parallel=dict(
            zero1=dict(size=-1, fsdp=False),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence_parallel=False,
            tensor=1,
        ),
        data=dict(
            seq_len=2048,
            micro_num=4,
            micro_bsz=2,
            pack_sample_into_one=False,
            min_length=50,
            total_steps=TOTAL_STEPS,
            valid_micro_num=4,
            valid_every=300,
            rampup_batch_size=None,
            diag_outlier_ratio=1.1,
            train_folder=os.path.join(
                os.environ["share_path"], "quailty_assurance/0623_scratch_tokenized_filtered/train"
            ),
            valid_folder=os.path.join(
                os.environ["share_path"], "quailty_assurance/0623_scratch_tokenized_filtered/val"
            ),
        ),
        model=dict(
            checkpoint=False,
            num_attention_heads=16,
            embed_split_hidden=True,
            vocab_size=103168,
            embed_grad_scale=1,
            parallel_output=True,
            hidden_size=4096,
            num_layers=16,
            mlp_ratio=8 / 3,
            apply_post_layer_norm=False,
            dtype="torch.bfloat16",
            norm_type="rmsnorm",
            layer_norm_epsilon=1e-5,
            use_flash_attn=True,
            num_chunks=1,
        ),
        model_type="INTERNLM",
        alert_address=None,
        monitor=dict(alert=dict(enable_feishu_alert=False, feishu_alert_address=None, light_monitor_address=None)),
        grad_scaler=dict(
            fp16=dict(
                initial_scale=2**16,
                min_scale=1,
                growth_interval=1000,
            ),
            growth_factor=2,
            backoff_factor=0.5,
            max_scale=2**24,
            hysteresis=2,
        ),
        adam=dict(
            lr=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_beta2_c=0,
            adam_eps=1e-8,
            weight_decay=0.01,
        ),
        hybrid_zero_optimizer=dict(
            overlap_sync_grad=True,
            overlap_sync_param=True,
            reduce_bucket_size=512 * 1024 * 1024,
            clip_grad_norm=1.0,
        ),
        beta2_scheduler=dict(
            init_beta2=0.95,
            c=0,
            cur_iter=-1,
        ),
        lr_scheduler=dict(
            total_steps=TOTAL_STEPS,
            init_steps=0,
            warmup_ratio=0.01,
            eta_min=1e-5,
            last_epoch=-1,
        ),
        ckpt=dict(
            enable_save_ckpt=False,
            auto_resume=False,
        ),
        loss=dict(
            label_smoothing=0,
        ),
    )
)


def build_environment(rank, world_size, config):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "33333"
    torch.cuda.empty_cache()
    # launcher="torch"
    internlm.launch_from_torch(config=config, seed=1024)
    args_sanity_check()


def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_new_batch(train_dl, train_iter):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dl)
        batch = next(train_iter)

    return batch, train_iter


def evaluate_on_val_dls(
    trainer,
    val_dls,
):
    torch.cuda.empty_cache()
    trainer.eval()
    verbose = gpc.is_rank_for_log()
    data_cfg = gpc.config.data

    for _, val_dl in val_dls.items():
        if len(val_dl) == 0 and verbose:
            continue

        val_metric = AccPerplex(
            device=torch.cuda.current_device(),
            tp_pg=gpc.get_group(ParallelMode.TENSOR),
            dp_pg=gpc.get_group(ParallelMode.DATA),
        )
        val_sche_metric_hook = SchedulerMetricHook(metric=val_metric)

        val_loss = 0
        val_idx = -1
        for val_idx, batch in tqdm(
            enumerate(val_dl),
            desc="Val.",
            total=len(val_dl),
            position=1,
            disable=not verbose,
            leave=False,
        ):
            with torch.inference_mode():
                total_val_bsz = len(batch[1])
                assert total_val_bsz % data_cfg.micro_bsz == 0
                grad_accum_size = total_val_bsz // data_cfg.micro_bsz
                with switch_evaluation_no_pipeline_scheduler(
                    trainer=trainer,
                    grad_accum_size=grad_accum_size,
                    metric_hook_list=[val_sche_metric_hook],
                ):
                    _, _, loss = trainer.execute_schedule(
                        batch, forward_only=True, return_loss=True, return_output_label=False
                    )

            if verbose:
                val_loss += loss.item()

        assert val_idx != -1
        dist.barrier()

        if verbose and len(val_dl) != 0:
            val_loss = val_loss / (val_idx + 1 + 1e-6)

    trainer.train()
    torch.cuda.empty_cache()
    dist.barrier()
    return val_loss


def compute_trimmed_mean(value_list):
    trim = int(0.05 * len(value_list))
    trimmed_list = value_list[trim:-trim]
    trimmed_mean = sum(trimmed_list) / len(trimmed_list)
    return trimmed_mean


def check_grad_norm(grad_norm_list):
    standard_grad_norm_list = torch.load(os.path.join(
        os.environ["share_path"], "quailty_assurance/small_300step_norm_grad/grad_norm_list.pt"
    ))
    
    standard_grad_norm_list = standard_grad_norm_list[-100:]
    grad_norm_list = grad_norm_list[-100:]
    standard_grad_norm_list.sort()
    grad_norm_list.sort()

    trimmed_mean1 = compute_trimmed_mean(standard_grad_norm_list)
    trimmed_mean2 = compute_trimmed_mean(grad_norm_list)
    tensor_trimmed_mean1 = torch.tensor(trimmed_mean1)
    tensor_trimmed_mean2 = torch.tensor(trimmed_mean2)
    
    logger.info(f"norm_mean: {tensor_trimmed_mean1}, {tensor_trimmed_mean2}")
    assert torch.allclose(tensor_trimmed_mean1, tensor_trimmed_mean2, rtol=3e-1, atol=3e-1)
    logger.info(f"grad norm check passed")
    

def check_meanLoss_val(all_loss, all_val):
    loss_values1 = all_loss[0][-100:]
    loss_values2 = all_loss[1][-100:]
    loss_values1.sort()
    loss_values2.sort()
    
    trimmed_mean1 = compute_trimmed_mean(loss_values1)
    trimmed_mean2 = compute_trimmed_mean(loss_values2)
    tensor_trimmed_mean1 = torch.tensor(trimmed_mean1)
    tensor_trimmed_mean2 = torch.tensor(trimmed_mean2)

    logger.info(f"avg_value: {trimmed_mean1}, {trimmed_mean2}")
    logger.info(f"all_val: {all_val}")

    assert torch.allclose(tensor_trimmed_mean1, tensor_trimmed_mean2, rtol=3e-2, atol=3e-2)
    assert torch.allclose(torch.tensor(all_val[0]), torch.tensor(all_val[1]), rtol=3e-2, atol=3e-2)
    
    logger.info(f"loss check passed")
    

def exam_loss(args):
    # init
    rank, world_size, micro_num, micro_bsz = args
    config.data.micro_num = micro_num
    config.data.micro_bsz = micro_bsz
    build_environment(rank, world_size, config)

    total_steps = gpc.config.data.total_steps
    valid_every = gpc.config.data.valid_every

    # set seed
    seed_all(1024)

    # initialize model
    model = initialize_model()

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=gpc.config.loss.label_smoothing)

    # initialize the train and validation data loader
    train_dl, dataset_types = get_train_data_loader(num_worker=0)
    val_dls = get_validation_data_loader()

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=model)

    # initialize metric for calculating accuracy and perplexity
    metric = AccPerplex(
        device=torch.cuda.current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
        dataset_types=dataset_types,
    )

    # initialize trainer
    scheduler_hooks = [
        SchedulerMetricHook(
            metric=metric,
            skip=(
                gpc.is_using_pp()
                and hasattr(gpc.config.model, "num_chunks")
                and gpc.config.model.num_chunks > 1
                and gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
            ),
        ),
    ]

    trainer, train_dl, _, _ = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dl,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=scheduler_hooks,
    )

    trainer.train()
    train_iter = iter(train_dl)

    # transfer the train data loader into train data iterator
    loss_list = []
    val_list = []
    grad_norm_list = []
    for batch_count in range(total_steps):
        start_time = time.time()
        # load batch data
        batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter)

        # zero the grads of parameters
        trainer.zero_grad()

        # process data
        if batch[0].get("type_ids", None) is not None:
            metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))

        _, _, loss = trainer.execute_schedule(
            batch,
            forward_only=False,
            return_loss=True,
            return_output_label=False,
        )
        loss_list.append(loss.item())

        num_tokens_in_batch = batch[1].nelement()
        tgs_origin = round(
            num_tokens_in_batch
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / (time.time() - start_time),
            2,
        )

        if rank == 0:
            logger.info(f"batch_count: {batch_count}, tgs: {tgs_origin}, loss: {loss}")

        # update parameters
        trainer_result = trainer.step()
        assert trainer_result is not None
        
        _, grad_norm_groups = trainer_result
        
        if gpc.is_rank_for_log():
            logger.info(f"train_grad_norm_groups: {grad_norm_groups['0_default']}")
            grad_norm_list.append(grad_norm_groups['0_default'])

        # evaluate on validation data loaders
        if valid_every > 0 and batch_count > 0 and (batch_count + 1) % valid_every == 0:
            val_result = evaluate_on_val_dls(
                trainer=trainer,
                val_dls=val_dls,
            )
            if val_result != 0:
                val_list.append(val_result)

    torch.cuda.empty_cache()
    dist.barrier()
    
    if gpc.is_rank_for_log():
        check_grad_norm(grad_norm_list)
    
    return rank, loss_list, val_list


def test_loss():
    ctx = mp.get_context("spawn")
    all_loss = []
    all_val = []
    micro_num = 4
    micro_bsz = 2
    for train_round in range(2):
        if train_round == 1:
            micro_num, micro_bsz = micro_bsz, micro_num
        with ctx.Pool(processes=8) as pool:
            results = pool.map(
                exam_loss,
                [[rank, 8, micro_num, micro_bsz] for rank in range(8)],
            )
            all_loss.append(results[0][1])
            all_val.append(results[0][2])
            pool.close()
            pool.join()

    check_meanLoss_val(all_loss, all_val)


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_swap_nb_loss_and_gradnorm.py"])
