#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import socket
import time
import traceback
from functools import partial
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import internlm
from internlm.core.amp import convert_to_amp
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.scheduler import SchedulerMetricHook
from internlm.core.trainer import TrainState
from internlm.data.batch_sampler import StaticBatchSampler, get_dpsampler_dataloader
from internlm.data.collaters import jsonl_ds_collate_fn, packed_collate_fn
from internlm.data.dataset import get_dataset_dict
from internlm.data.dummy_dataset import RandomDataset
from internlm.data.packed_dataset import (
    PackedDataset,
    PackedDatasetWithoutCuSeqlen,
    get_packed_dataset_without_short_length,
)
from internlm.data.utils import DATASET_TYPE_IDS_MAP, unpack_data
from internlm.model.loss import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex
from internlm.monitor import initialize_monitor_manager, send_alert_message, set_env_var
from internlm.monitor.monitor import monitor_manager as mm
from internlm.solver.beta2_scheduler import Beta2Scheduler
from internlm.solver.lr_scheduler import FineTuneCosineAnnealingWarmupLR
from internlm.solver.optimizer import HybridZeroOptimizer
from internlm.utils.common import (
    BatchSkipper,
    DummyProfile,
    get_master_node,
    get_megatron_flops,
    launch_time,
    parse_args,
)
from internlm.utils.evaluation import evaluate_on_val_dls
from internlm.utils.logger import get_logger, initialize_uniscale_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.model_checkpoint import CheckpointManager
from internlm.utils.parallel import (
    get_parallel_log_file_name,
    is_no_pp_or_last_stage,
    sync_model_param,
    sync_model_param_within_tp,
)
from internlm.utils.registry import MODEL_INITIALIZER
from internlm.utils.simple_memory_profiler import SimpleMemoryProfiler
from internlm.utils.writer import Writer


# global llm logger
logger = get_logger(__file__)


def initialize_distributed_env(config: str, launcher: str = "slurm", master_port: int = 8888, seed: int = 1024):
    """
    Initialize distributed environment for distributed training.

    Args:
        config (str): Config file path.
        launcher (str): Launcher for launching distributed environment, can be slurm or torch. "slurm" by default.
        master_port (str): The master port for distributed training. 8888 by default.
        seed (int, optional): Specified random seed for every process. 1024 by default.
    """

    torch.cuda.empty_cache()

    if launcher == "torch":
        internlm.launch_from_torch(config=config, seed=seed)
    elif launcher == "slurm":
        internlm.launch_from_slurm(
            config=config,
            host=get_master_node(),
            port=master_port,
            seed=seed,
        )
    else:
        assert launcher in ["slurm", "torch"], "launcher only support slurm or torch"


def initialize_llm_logger(start_time: str):
    """
    Initialize customed uniscale logger.

    Args:
        start_time (str): The launch time of current training job.

    Returns: The instance of uniscale logger.
    """

    uniscale_logger = initialize_uniscale_logger(
        job_name=gpc.config.JOB_NAME, launch_time=start_time, file_name=get_parallel_log_file_name()
    )
    if uniscale_logger is not None:
        global logger
        logger = uniscale_logger

    return uniscale_logger


def initialize_model(criterion):
    """
    Initialize model.

    Returns: The neural network model to be trained or evaluated.
    """

    model = MODEL_INITIALIZER.get_module(module_name=gpc.config.model_type)(**(gpc.config.model))
    
    model, criterion = convert_to_amp(model, criterion, gpc.config.model.use_amp)

    # This sync is very important, cause the model weights kept in optimizer are copied
    # from the origin parameters in the memory, so we should make sure the dp sync
    # does not influence the model weights in optimizer be different with the origin parameters.
    sync_model_param(model, parallel_mode=ParallelMode.DATA)

    # This function is needed to make sure parameters that are not splitted by tensor parallelism are
    # the same across tensor parallelism.
    sync_model_param_within_tp(model)
    
    # def get_intermediate_output(module, input, output):
    #     print(module, output.dtype)
    
    # def bwd_hook(module, grad_input, grad_output):
    #     import pdb; pdb.set_trace()
    #     print("bwd: ", end='')
    #     r = []
    #     if isinstance(grad_input, torch.Tensor) and isinstance(grad_output, torch.Tensor):
    #         print(module, grad_input.dtype, grad_output.dtype)
    #         return
    #     for t in grad_input:
    #         if t is None:
    #             continue
    #         r.append(t.dtype)
    #     for t in grad_output:
    #         if t is None:
    #             continue
    #         r.append(t.dtype)
    #     print(module, r)
    
    # def add_hooks_recursively(module):
    #     # module.register_forward_hook(get_intermediate_output)
    #     module.register_full_backward_hook(bwd_hook)
        
    #     for child in module.children():
    #         add_hooks_recursively(child)
    
    # add_hooks_recursively(model)

    return model, criterion


def get_train_data_loader(num_worker: int = 0):
    """
    Generate and return the training data loader.

    Returns: A tuple of (train_dl, dataset_types).
    """

    # Get the dataset types
    dataset_types = None
    dataset_types = list(DATASET_TYPE_IDS_MAP.keys())
    data_cfg = gpc.config.data

    # Get the sample weight dictionary
    train_folder = data_cfg.train_folder

    if not train_folder:
        train_ds = RandomDataset(num_samples=1000000, max_len=data_cfg.seq_len)
        if data_cfg.pack_sample_into_one:
            train_ds = PackedDatasetWithoutCuSeqlen(
                train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
            )
        else:
            train_ds = PackedDataset(
                train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
            )
    else:
        train_ds = get_packed_dataset_without_short_length(
            folder=data_cfg.train_folder,
            packed_length=data_cfg.packed_length,
            max_length_per_sample=data_cfg.seq_len,
            show_progress=dist.get_rank() == 0,
            min_length=data_cfg.min_length,
            min_length_dict=data_cfg.get("min_length_dict", {}),
            pack_into_one_sample=data_cfg.pack_sample_into_one,
        )

    # partition already completed
    # assert isinstance(train_ds, (PackedDataset, PackedDatasetWithoutCuSeqlen))
    if isinstance(train_ds, (PackedDataset, PackedDatasetWithoutCuSeqlen)):
        datasets = [train_ds]
    else:
        datasets = train_ds.datasets

    # Create the training dataset sampler
    train_sampler = StaticBatchSampler(
        datasets,
        batch_size=data_cfg.micro_num,
        rampup_batch_size=data_cfg.rampup_batch_size,
        micro_bsz=data_cfg.micro_bsz,
        seed=1024,
        drop_last=True,
        data_rank=gpc.get_local_rank(ParallelMode.DATA),
        data_world_size=gpc.get_world_size(ParallelMode.DATA),
    )

    train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)

    # Create the training data loader
    train_dl = DataLoader(
        dataset=train_ds,
        batch_sampler=train_sampler,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=train_collate_fn,
        persistent_workers=True,
    )

    return train_dl, dataset_types


def get_validation_data_loader(num_worker: int = 0):
    """Generate and return the validation data loader."""

    data_cfg = gpc.config.data

    if not data_cfg.valid_folder:
        val_ds = RandomDataset(num_samples=gpc.get_world_size(ParallelMode.DATA) * 500, max_len=data_cfg.seq_len)
    else:
        val_ds = get_dataset_dict(folder=data_cfg.valid_folder, split="")

    if not isinstance(val_ds, dict):
        val_ds = {"val": val_ds}

    val_collate_fn = partial(jsonl_ds_collate_fn, max_length_per_sample=data_cfg.seq_len)

    val_dls = {}
    for val_name, ds in val_ds.items():
        # making the batch_size of validate larger can speed up the evaluation, but it should not be too large,
        # otherwise too much data may be dropped
        batch_size = min(
            data_cfg.valid_micro_num * data_cfg.micro_bsz, len(ds) // gpc.get_world_size(ParallelMode.DATA)
        )
        batch_size = batch_size // data_cfg.micro_bsz * data_cfg.micro_bsz

        if batch_size == 0 and gpc.is_rank_for_log():
            logger.info(f"skip validate {val_name}.")
            continue

        val_dls[val_name] = get_dpsampler_dataloader(
            ds, shuffle=False, num_workers=num_worker, batch_size=batch_size, collate_fn=val_collate_fn, drop_last=True
        )  # drop_last=True, otherwise it may cause problems in the last batch

        if gpc.is_rank_for_log():
            logger.info(
                f"load validation dataset {val_name} with valid batch size {str(batch_size)} and "
                f"samples {str(len(val_dls[val_name]))}."
            )

    return val_dls


def load_new_batch(train_dl: DataLoader, train_iter: Iterable, train_state: TrainState):
    """
    Load and return the new batch data based on training data loader.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader for training.
        train_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
        train_state (TrainState): Current training state.

    Returns: A batch data and the updated train_iter.
    """

    timer("batch-gen").start()
    try:
        batch = next(train_iter)  # structure is ({'input_ids': Tensor, 'cu_seqlens': Tensor}, Tensor)
        next(train_state.batch_sampler_iter)
    except StopIteration:
        train_iter = iter(train_dl)
        batch = next(train_iter)
        train_state.batch_sampler_iter = iter(train_state.batch_sampler)
        next(train_state.batch_sampler_iter)
        train_state.num_consumed_samples_in_epoch = 0
    timer("batch-gen").stop()

    if batch[0].get("type_ids", None) is not None:
        # if use_flash_attn is False, we need to unpack type_ids
        if not gpc.config.model.use_flash_attn:
            batch[0]["type_ids"] = unpack_data(batch[0]["type_ids"], batch[0]["cu_seqlens"])

    return batch, train_iter


def initialize_optimizer(model: nn.Module):
    """
    Initialize optimizer.

    Args:
        model (torch.nn.Module): Your model instance to be trained or evaluated.

    Returns: A tuple of (optimizer, beta2_scheduler, lr_scheduler).
    """
    adam_cfg = gpc.config.adam
    naive_optimizer = torch.optim.AdamW(
        params=[{"params": model.parameters(), "weight_decay": adam_cfg.weight_decay}],
        lr=adam_cfg.lr,
        betas=(adam_cfg.adam_beta1, adam_cfg.adam_beta2),
        eps=adam_cfg.adam_eps,
    )

    optimizer = HybridZeroOptimizer(
        naive_optimizer, grad_scal_cfg=gpc.config.grad_scaler, zero_cfg=gpc.config.hybrid_zero_optimizer
    )

    beta2_scheduler = Beta2Scheduler(optimizer=naive_optimizer, **gpc.config.beta2_scheduler)

    lr_scheduler = FineTuneCosineAnnealingWarmupLR(optimizer, **gpc.config.lr_scheduler)

    return optimizer, beta2_scheduler, lr_scheduler


def initialize_llm_profile(profiling: bool = False, start_time: str = None):
    """Initialize and return the profiler context manager instance."""

    if profiling and gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        llm_profile = torch.profiler.profile
        logger.info(f"Do profiling in rank {gpc.get_global_rank()}!")
    else:
        llm_profile = DummyProfile

    return llm_profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=5, wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"{gpc.config.JOB_NAME}/{start_time}/traces/rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"pp{gpc.get_local_rank(ParallelMode.PIPELINE)}",
        ),
        with_stack=True,
        with_modules=True,
    )


def record_current_batch_training_metrics(
    get_tflops_func,
    logger,
    writer,
    success_update,
    batch_count,
    batch,
    train_state,
    optimizer,
    beta2_scheduler,
    trainer,
    start_time,
    loss,
    grad_norm,
    metric,
    update_panel,
):
    """
    Print some training metrics of current batch.
    """

    set_env_var(key="LAST_ACTIVE_TIMESTAMP", value=int(time.time()))

    if success_update in (0, True):
        train_state.num_consumed_tokens += batch[1].nelement() * gpc.get_world_size(ParallelMode.DATA)
    if is_no_pp_or_last_stage():
        acc_perplex = metric.get_metric()

    if success_update and gpc.is_rank_for_log():
        lr = optimizer.param_groups[0]["lr"]
        if hasattr(trainer.engine.optimizer, "grad_scaler"):
            scaler = trainer.engine.optimizer.grad_scaler._scale.item()
        elif hasattr(trainer.engine.optimizer.optim, "grad_scaler"):
            scaler = trainer.engine.optimizer.optim.grad_scaler._scale.item()

        num_tokens_in_batch = batch[1].nelement()
        num_samples_in_batch = sum([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        max_length_in_batch = max([(b[1:] - b[:-1]).max().item() for b in batch[0]["cu_seqlens"]])
        max_samples_in_batch = max([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        min_samples_in_batch = min([len(b) - 1 for b in batch[0]["cu_seqlens"]])

        tk_per_gpu = 0
        tk_per_gpu = round(
            num_tokens_in_batch
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / (time.time() - start_time),
            2,
        )

        tflops = get_tflops_func((time.time() - start_time))

        infos = {
            "tflops": tflops,
            "step": batch_count,
            "loss": loss.item(),
            "tgs (tokens/gpu/second)": tk_per_gpu,
            "lr": lr,
            "loss_scale": scaler,
            "grad_norm": grad_norm,
        }

        infos["micro_num"] = len(batch[1])
        infos["num_consumed_tokens"] = train_state.num_consumed_tokens
        infos["inf_nan_skip_batches"] = train_state.inf_nan_skip_batches
        infos["num_samples_in_batch"] = num_samples_in_batch  # the number of batches which have the most samples
        infos["largest_length"] = max_length_in_batch  # the longest input
        infos["largest_batch"] = max_samples_in_batch  # the batch with the most samples
        infos["smallest_batch"] = min_samples_in_batch
        infos["adam_beta2"] = beta2_scheduler.get_beta2()

        fwd_bwd_time = round(timer("fwd-bwd").elapsed(), 2)
        infos["fwd_bwd_time"] = fwd_bwd_time

        for key, value in acc_perplex.items():
            infos[key] = value

        line = ""
        for key, value in infos.items():
            line += f"{key}={value} "
            writer.add_scalar(key=key, value=value, step=train_state.step_count)

        if update_panel:
            logger.info(
                line,
                extra={
                    "step": batch_count,
                    "lr": lr,
                    "num_consumed_tokens": train_state.num_consumed_tokens,
                    "grad_norm": grad_norm,
                    "loss": loss.item(),
                    "flops": tflops,
                    "tgs": tk_per_gpu,
                    "acc": acc_perplex["acc"],
                    "perplexity": acc_perplex["perplexity"],
                    "fwd_bwd_time": fwd_bwd_time,
                },
            )
        else:
            logger.info(line)

        # if loss spike occurs, send alert info to feishu
        mm.monitor_loss_spike(alert_address=gpc.config.alert_address, step_count=batch_count, cur_step_loss=loss.item())


def main(args):
    # init setting
    skip_batches = gpc.config.data.skip_batches
    total_steps = gpc.config.data.total_steps
    valid_every = gpc.config.data.valid_every
    label_smoothing = gpc.config.loss.label_smoothing
    lr = gpc.config.adam.lr

    get_tflops_func = partial(
        get_megatron_flops,
        checkpoint=gpc.config.model.checkpoint,
        seq_len=gpc.config.SEQ_LEN,
        hidden_size=gpc.config.model.hidden_size,
        num_layers=gpc.config.model.num_layers,
        vocab_size=gpc.config.model.vocab_size,
        global_batch_size=gpc.config.data.micro_bsz * gpc.config.data.micro_num * gpc.get_world_size(ParallelMode.DATA),
        global_world_size=gpc.get_world_size(ParallelMode.GLOBAL),
        mlp_ratio=gpc.config.MLP_RATIO,
    )

    # get and broadcast current time
    current_time = launch_time()
    objs = [current_time]
    dist.broadcast_object_list(objs, src=0)
    current_time = objs[0]

    # initialize customed llm logger
    uniscale_logger = initialize_llm_logger(start_time=current_time)

    # initialize customed llm writer
    with open(args.config, "r") as f:
        config_lines = f.readlines()
    writer = Writer(
        job_name=gpc.config.JOB_NAME,
        launch_time=current_time,
        file_name=get_parallel_log_file_name(),
        tensorboard_folder=gpc.config.tensorboard_folder,
        resume_tb_folder=gpc.config.resume_tb_folder,
        config=config_lines,
        logger=logger,
        enable_tb=gpc.config.enable_tb,
    )

    # initialize and resume train state
    train_state = TrainState(gpc.config)

    ckpt_manager = CheckpointManager(
        ckpt_config=gpc.config.ckpt,
        model=model,
        model_config=gpc.config.model,
        feishu_address=gpc.config.alert_address,
    )

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=label_smoothing)
    
    # initialize model
    model, criterion = initialize_model(criterion)

    # initialize the train and validation data loader
    train_dl, dataset_types = get_train_data_loader(num_worker=4)
    val_dls = get_validation_data_loader()
    train_state.init_batch_sampler(train_dl)

    # Loading model weights must be done before zero is initialized.
    ckpt_manager.try_load_model(current_time)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=model)

    # Loading other persistent training states.
    ckpt_manager.try_resume_training(lr_scheduler, optimizer, lr, train_state, train_dl)

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
    
     # initialize simple memory profiler
    if args.profiling:
        memory_profiler = SimpleMemoryProfiler(
            model,
            optimizer.optim,
            log_folder=f"memory_trace/rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}",
        )
    else:
        memory_profiler = None

    # initialize simple memory profiler
    if args.profiling:
        memory_profiler = SimpleMemoryProfiler(
            model,
            optimizer.optim,
            log_folder=f"memory_trace/rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}",
        )
    else:
        memory_profiler = None

    # initialize the batch skipper
    batch_skipper = BatchSkipper(skip_batches)

    trainer.train()

    # transfer the train data loader into train data iterator
    train_iter = iter(train_dl)

    with initialize_llm_profile(profiling=args.profiling, start_time=current_time) as prof:
        # start iterating the train data and begin training
        for batch_count in range(train_state.batch_count, total_steps):
            if batch_count % 50 == 0:
                torch.cuda.empty_cache()

            start_time = time.time()
            timer("one-batch").start()

            # load batch data
            batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter, train_state=train_state)

            # record the consumed samples in training
            train_state.batch_count = batch_count
            train_state.num_consumed_samples_in_epoch += len(batch[1])
            if batch_skipper(batch_count):  # skip this batch
                if gpc.is_rank_for_log():
                    logger.info(f"Skip batch count:`{batch_count}`...")
                timer("one-batch").stop()
                continue

            # zero the grads of parameters
            trainer.zero_grad()
            # process data
            if batch[0].get("type_ids", None) is not None:
                metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))
            
            # do forward and backward
            timer("fwd-bwd").start()
            _, _, loss = trainer.execute_schedule(
                batch, forward_only=False, return_loss=True, return_output_label=False
            )
            timer("fwd-bwd").stop()
            
            # update parameters, and returns (success_update, grad_norm)
            trainer_result = trainer.step()
            assert trainer_result is not None

            success_update, grad_norm_groups = trainer_result
            if success_update:  # update parameters successfully
                train_state.step_count += 1
            else:
                train_state.inf_nan_skip_batches += 1  # record the amount of updating parameters unsuccessfully.
                if -99.0 in grad_norm_groups and gpc.is_rank_for_log():  # -99.0 encodes a specific failure case
                    logger.warning(f"Warning: skip parameter update at step {batch_count}.")
                    send_alert_message(
                        address=gpc.config.alert_address,
                        message=f"Warning: skip parameter update at step {batch_count}.",
                    )

            # calculate and record the training metrics, eg. loss, accuracy and so on.
            record_current_batch_training_metrics(
                get_tflops_func=get_tflops_func,
                logger=logger,
                writer=writer,
                success_update=success_update,
                batch_count=batch_count,
                batch=batch,
                train_state=train_state,
                optimizer=optimizer,
                beta2_scheduler=beta2_scheduler,
                trainer=trainer,
                start_time=start_time,
                loss=loss,
                grad_norm=np.array(grad_norm_groups),
                metric=metric,
                update_panel=uniscale_logger is not None,
            )

            timer("one-batch").stop()

            # evaluate on validation data loaders
            if valid_every > 0 and train_state.step_count % valid_every == 0:
                evaluate_on_val_dls(
                    trainer=trainer,
                    val_dls=val_dls,
                    writer=writer,
                    logger=logger,
                    step_count=train_state.step_count,
                    update_panel=uniscale_logger is not None,
                )

            # checkpoint the training states in specific steps, which is determined by the args "checkpoint_every"
            # # save batch sampler that tracks the true consumed samples
            now_break = ckpt_manager.try_save_checkpoint(train_state)
            if now_break:
                break

            if memory_profiler is not None:
                memory_profiler.step()

            prof.step()
            
            torch.cuda.reset_max_memory_allocated()

    ckpt_manager.wait_async_upload_finish()


if __name__ == "__main__":
    args = parse_args()
    hostname = socket.gethostname()

    # initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # initialize monitor manager context
    with initialize_monitor_manager(job_name=gpc.config.JOB_NAME, alert_address=gpc.config.alert_address):
        try:
            main(args)
        except Exception as e:
            # logger.error(
            #     f"Raise exception from {hostname} with rank id: {gpc.get_global_rank()}",
            #     exc_info=traceback.format_exc(),
            # )
            logger.error(
                f"Raise exception from {hostname} with rank id: {gpc.get_global_rank()}\n{traceback.format_exc()}",
            )
            logger.exception(e)
            mm.monitor_exception(alert_address=gpc.config.alert_address, excp_info=traceback.format_exc())
