import random

import numpy as np
import torch
from torch import nn
from torch.testing import assert_close

import internlm
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.core.gradient_handler import PipelineSharedModuleGradientHandler
from internlm.core.scheduler import (
    InterleavedPipelineScheduler,
    NonPipelineScheduler,
    PipelineScheduler,
    SchedulerMetricHook,
)
from internlm.solver.pipeline_utils import partition_uniform
from internlm.train import initialize_optimizer


class MlpModel(nn.Module):
    """
    Custom model
    """

    def __init__(self, start, end, model_type=None, embedding=False):
        super().__init__()
        self.part = [start, end]
        self.blocks = nn.ModuleList([nn.Linear(8, 8, bias=False) for lid in range(end - start)])
        self.model_type = model_type
        self.embedding = embedding

    def forward(
        self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None, **kwargs
    ):  # pylint: disable=W0613
        if self.model_type != "torch" and self.part[0] != 0:
            input_ids = hidden_states

        # Simulate Embedding.
        if self.embedding:
            if len(input_ids.shape) == 2:
                input_ids = input_ids.view(-1, 8)
            elif len(input_ids.shape) == 3:
                input_ids = input_ids.view(input_ids.shape(0), -1, 8)

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


def init_model_and_optim(
    num_layers, num_chunks, dtype, micro_num, interleaved_overlap, tensor_shape, init_optim=True, embedding=False
):
    # pp model
    pp_model = _build_generic_model_1d(num_layers=num_layers, num_chunks=num_chunks, embedding=embedding)
    pp_model = pp_model.to(dtype)

    # pp scheduler
    scheduler_hooks = [
        SchedulerMetricHook(skip=True),
    ]

    if gpc.get_world_size(ParallelMode.PIPELINE) > 1:
        if num_chunks == 1:
            # noninterleaved pp
            scheduler = PipelineScheduler(
                data_process_func=None,
                num_microbatches=micro_num,
                dtype=dtype,
                tensor_shape=tensor_shape,
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
                        tensor_shape=tensor_shape,
                        scatter_gather_tensors=False,
                        scheduler_hooks=scheduler_hooks,
                        communication_overlap=interleaved_overlap,
                    )
                except AssertionError as e:
                    print(f"AssertionError: {e}", flush=True)
                    return None, None
                else:
                    raise RuntimeError(
                        "Error: AssertionError should occur when micro_num < Pipeline parrallel world size"
                    )
            else:
                scheduler = InterleavedPipelineScheduler(
                    num_microbatches=micro_num,
                    num_chunks=gpc.config.model.num_chunks,
                    dtype=dtype,
                    tensor_shape=tensor_shape,
                    scatter_gather_tensors=False,
                    scheduler_hooks=scheduler_hooks,
                    communication_overlap=interleaved_overlap,
                )
    else:
        scheduler = NonPipelineScheduler(
            data_process_func=None,
            gradient_accumulation_size=gpc.config.data.gradient_accumulation,
            scheduler_hooks=scheduler_hooks,
        )

    # pp optimizer and engine
    if init_optim:
        optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=pp_model)
    else:
        optimizer, beta2_scheduler, lr_scheduler = None, None, None

    engine = Engine(
        model=pp_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        criterion=MyLoss().to(dtype),
        gradient_handlers=[PipelineSharedModuleGradientHandler(model=pp_model, optimizer=optimizer)],
        clip_grad_norm=0.0,
    )
    return engine, scheduler


def build_environment(rank, world_size, config):
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


def _build_generic_model_1d(num_layers, num_chunks, embedding=False):
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]
    if gpc.is_rank_for_log():
        print(f"The layer sharding is {all_parts}.", flush=True)

    models = []
    for start, end in parts:
        models.append(MlpModel(start, end, embedding=embedding).cuda())
    torch.distributed.barrier()
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model
