import multiprocessing as mp
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.testing import assert_close

import internlm
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config
from internlm.core.engine import Engine
from internlm.core.gradient_handler import PipelineSharedModuleGradientHandler
from internlm.core.scheduler import (
    InterleavedPipelineScheduler,
    PipelineScheduler,
    SchedulerMetricHook,
)
from internlm.solver.pipeline_utils import partition_uniform
from internlm.train import initialize_optimizer


class MlpModel(nn.Module):
    """
    Custom model
    """

    def __init__(self, start, end, model_type=None):
        super().__init__()
        self.part = [start, end]
        self.blocks = nn.ModuleList([nn.Linear(8, 8, bias=False) for lid in range(end - start)])
        self.model_type = model_type

    def forward(self, hidden_states=None, input_ids=None):
        if self.model_type != "torch" and self.part[0] != 0:
            input_ids = hidden_states

        for i in range(self.part[1] - self.part[0]):
            input_ids = self.blocks[i](input_ids)
        return input_ids


class MyLoss(nn.Module):
    """
    Custom loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        loss = torch.nn.MSELoss(reduction="sum")
        return loss(logits, labels)


config = Config(
    dict(
        gradient_handler=[dict(type="PipelineSharedModuleGradientHandler")],
        parallel=dict(
            zero1=dict(size=1, fsdp=False),
            pipeline=dict(size=8, interleaved_overlap=False),
            sequence_parallel=False,
            tensor=1,
        ),
        model_type="INTERNLM",
        data=dict(seq_len=8, micro_num=16, micro_bsz=1, pack_sample_into_one=False, min_length=0, total_steps=9999),
        model=dict(
            dtype=torch.bfloat16,
            num_chunks=2,
            use_flash_attn=True,
        ),
        resume_tb_folder="",
        tensorboard_folder="",
        alert_address=None,
        monitor=dict(alert=dict(enable_feishu_alert=False, feishu_alert_address=None, light_monitor_address=None)),
        grad_scaler=dict(
            fp16=dict(
                initial_scale=1,
                min_scale=1,
                growth_interval=1,
            ),
            growth_factor=1.1,
            backoff_factor=0.9,
            max_scale=1,
            hysteresis=1,
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
            overlap_sync_grad=False,
            overlap_sync_param=False,
            reduce_bucket_size=512 * 1024 * 1024,
            clip_grad_norm=1.0,
        ),
        beta2_scheduler=dict(
            init_beta2=0.95,
            c=0,
            cur_iter=-1,
        ),
        lr_scheduler=dict(
            total_steps=100,
            init_steps=0,
            warmup_ratio=0.01,
            eta_min=1e-5,
            last_epoch=-1,
        ),
    )
)


def build_environment(rank, world_size):
    import os

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "33333"
    torch.cuda.empty_cache()
    # launcher="torch"
    internlm.launch_from_torch(config=config, seed=1024)


def loose_close(a, b, dtype: torch.dtype = torch.float32):

    if dtype is torch.float32:
        rtol = 1.3e-6
        atol = 1e-5
    elif dtype is torch.bfloat16:
        rtol = 2e-2
        atol = 2e-2

    if isinstance(a, torch.Tensor):
        a = a.detach().to(dtype)
        b = b.detach().to(dtype)

    assert_close(a, b, rtol=rtol, atol=atol)


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


def _build_generic_model_1d(num_layers, num_chunks):
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]
    if gpc.is_rank_for_log():
        print(f"The layer sharding is {all_parts}.", flush=True)

    models = []
    for start, end in parts:
        models.append(MlpModel(start, end).cuda())
    torch.distributed.barrier()
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model


def exam_pipeline_parallel(args):
    # init
    rank, world_size, micro_num, num_chunks, interleaved_overlap = args
    config.data.micro_num = micro_num
    config.model.num_chunks = num_chunks
    config.parallel.pipeline.interleaved_overlap = interleaved_overlap

    build_environment(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dtype = config.model["dtype"]

    # set seed
    seed_all(1024)

    # pp model
    pp_model = _build_generic_model_1d(num_layers=32, num_chunks=num_chunks)
    pp_model = pp_model.to(dtype)

    # pp scheduler
    scheduler_hooks = [
        SchedulerMetricHook(skip=True),
    ]

    seq_len = gpc.config.data.seq_len
    gpc.config.NUM_MICRO_BATCHES = micro_num
    communication_overlap = interleaved_overlap

    if num_chunks == 1:
        # noninterleaved pp
        scheduler = PipelineScheduler(
            data_process_func=None,
            num_microbatches=micro_num,
            dtype=dtype,
            tensor_shape=[1, 8],
            scatter_gather_tensors=False,
            scheduler_hooks=scheduler_hooks,
        )
    else:
        # interleaved pp
        if micro_num < gpc.get_world_size(ParallelMode.PIPELINE):
            try:
                scheduler = InterleavedPipelineScheduler(
                    num_microbatches=micro_num,
                    num_chunks=gpc.config.model.num_chunks,
                    dtype=dtype,
                    tensor_shape=[1, 8],
                    scatter_gather_tensors=False,
                    scheduler_hooks=scheduler_hooks,
                    communication_overlap=communication_overlap,
                )
            except AssertionError:
                return
            else:
                raise RuntimeError("Error: AssertionError should occur when micro_num < Pipeline parrallel world size")
        else:
            scheduler = InterleavedPipelineScheduler(
                num_microbatches=micro_num,
                num_chunks=gpc.config.model.num_chunks,
                dtype=dtype,
                tensor_shape=[1, 8],
                scatter_gather_tensors=False,
                scheduler_hooks=scheduler_hooks,
                communication_overlap=communication_overlap,
            )

    # pp optimizer and engine
    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=pp_model)
    engine = Engine(
        model=pp_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        criterion=MyLoss().to(dtype),
        gradient_handlers=[PipelineSharedModuleGradientHandler(model=pp_model, optimizer=optimizer)],
        clip_grad_norm=gpc.config.hybrid_zero_optimizer.get("clip_grad_norm", 0.0),
    )

    scheduler.pre_processing(engine)
    engine.train()

    # create input
    x_list = []
    y_list = []
    for _ in range(micro_num):
        x_list.append(list(range(seq_len)))
        y_list.append(list(range(seq_len)))
    xs = torch.tensor(x_list).to(device).to(dtype)
    yx = torch.tensor(y_list).to(device).to(dtype)

    input_list = [{"input_ids": xs}, yx]

    # pp forward and backward
    output, _, loss = scheduler.forward_backward_step(
        engine, input_list, forward_only=False, return_loss=True, return_output_label=True
    )

    engine.step()

    # torch related
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        torch_xs = torch.tensor(x_list).to(device).to(torch.float32)
        torch_ys = torch.tensor(y_list).to(device).to(torch.float32)
        torch_model = MlpModel(0, 32, "torch").to(device)
        torch_optimizer = torch.optim.AdamW(
            params=[{"params": torch_model.parameters(), "weight_decay": config.adam.weight_decay}],
            lr=config.adam.lr,
            betas=(config.adam.adam_beta1, config.adam.adam_beta2),
            eps=config.adam.adam_eps,
        )

        # check output
        torch_output = torch_model(input_ids=torch_xs)  # pylint: disable=E1102
        loose_close(torch_output, output, dtype=dtype)

        torch_criterion = MyLoss().to(torch.float32)
        torch_loss = torch_criterion(torch_output, torch_ys) / micro_num  # pylint: disable=E1102
        torch_loss.backward()
        torch_optimizer.step()

        # check loss
        loose_close(torch_loss, loss[0], dtype=dtype)


@pytest.mark.parametrize("micro_num", [4, 8, 16])
@pytest.mark.parametrize("num_chunks", [1, 2, 4])
@pytest.mark.parametrize("interleaved_overlap", [True, False])
def test_pipeline_parallel(micro_num, num_chunks, interleaved_overlap):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(
            exam_pipeline_parallel,
            [[rank, 8, micro_num, num_chunks, interleaved_overlap] for rank in range(8)],
        )
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_pipeline.py"])
