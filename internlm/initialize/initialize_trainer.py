#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/initialize

from typing import Callable, Iterable, List, Optional, Tuple

from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.core.gradient_handler import PipelineSharedModuleGradientHandler
from internlm.core.scheduler import (
    InterleavedPipelineScheduler,
    NonPipelineScheduler,
    PipelineScheduler,
    SchedulerHook,
)
from internlm.core.scheduler.pipeline_scheduler import get_tensor_shape
from internlm.core.trainer import Trainer
from internlm.data.utils import unpack_data
from internlm.solver.beta2_scheduler import Beta2Scheduler
from internlm.solver.optimizer.hybrid_zero_optim import BaseOptimizer
from internlm.utils.common import get_current_device


def initialize_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Optional[_Loss] = None,
    train_dataloader: Optional[Iterable] = None,
    test_dataloader: Optional[Iterable] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    beta2_scheduler: Optional[Beta2Scheduler] = None,
    scheduler_hooks: Optional[List[SchedulerHook]] = None,
) -> Tuple[Trainer, DataLoader, DataLoader, _LRScheduler]:
    """Core function to wrap the essential training components with our functionality based on the config which is
    loaded into gpc.config.

    Args:
        model (:class:`torch.nn.Module` or `Callable`): Your model instance or a function to build the model.
        optimizer (:class:`BaseOptimizer`): Your optimizer for training.
        criterion (:class:`torch.nn.modules.loss._Loss`, optional): Your criterion instance.
        train_dataloader (:class:`torch.utils.data.DataLoader`, optional): Dataloader for training.
        test_dataloader (:class:`torch.utils.data.DataLoader`, optional): Dataloader for testing.
        lr_scheduler (:class:`torch.nn.lr_scheduler._LRScheduler`, optional): Your lr scheduler instance, optional.

    Returns:
        Tuple (trainer, train_dataloader, test_dataloader, lr_scheduler):
            A tuple of ``(trainer, train_dataloader, test_dataloader, lr_scheduler)``
            where only ``trainer`` could not be None.
    """

    if isinstance(model, nn.Module):
        # first sync model across dp ranks
        model.to(get_current_device())
    elif isinstance(model, Callable):
        model = model().to(get_current_device())

    # clip grad norm
    clip_grad_norm = gpc.config.hybrid_zero_optimizer.get("clip_grad_norm", 0.0)

    assert isinstance(optimizer, BaseOptimizer), "optimizer must be instance of BaseOptimizer"

    # gradient handler, only support PipelineSharedModuleGradientHandler now
    if gpc.is_using_pp():
        gpc.config.gradient_handler = [dict(type="PipelineSharedModuleGradientHandler")]
    gradient_handler_cfg = gpc.config.get("gradient_handler", [])
    gradient_handlers = []
    assert isinstance(gradient_handler_cfg, list), f"gradient_handler must be list but got {type(gradient_handler_cfg)}"
    for config in gradient_handler_cfg:
        if isinstance(config, dict) and config.get("type") == "PipelineSharedModuleGradientHandler":
            handler = PipelineSharedModuleGradientHandler(model=model, optimizer=optimizer)
            gradient_handlers.append(handler)

    # initialize scheduler for trainer
    scheduler = None
    if gpc.config.model.use_flash_attn:
        data_fn = None
    else:
        data_fn = unpack_data
    if gpc.is_using_pp():
        gpc.config.NUM_MICRO_BATCHES = gpc.config.data.micro_num
        tensor_shape = get_tensor_shape()
        use_interleaved = (
            hasattr(gpc.config, "model") and hasattr(gpc.config.model, "num_chunks") and gpc.config.model.num_chunks > 1
        )
        scatter_gather = gpc.is_initialized(ParallelMode.TENSOR)
        if use_interleaved:
            if isinstance(model, nn.Sequential):
                model = nn.ModuleList([model])

            communication_overlap = gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
            scheduler = InterleavedPipelineScheduler(
                num_microbatches=gpc.config.NUM_MICRO_BATCHES,
                num_chunks=gpc.config.model.num_chunks,
                dtype=gpc.config.model["dtype"],
                tensor_shape=tensor_shape,
                scatter_gather_tensors=scatter_gather,
                scheduler_hooks=scheduler_hooks,
                communication_overlap=communication_overlap,
            )
        else:
            scheduler = PipelineScheduler(
                data_process_func=data_fn,
                num_microbatches=gpc.config.NUM_MICRO_BATCHES,
                dtype=gpc.config.model["dtype"],
                tensor_shape=tensor_shape,
                scatter_gather_tensors=scatter_gather,
                scheduler_hooks=scheduler_hooks,
            )
    else:
        scheduler = NonPipelineScheduler(
            data_process_func=data_fn,
            gradient_accumulation_size=gpc.config.data.gradient_accumulation,
            scheduler_hooks=scheduler_hooks,
        )

    # initialize engine for trainer
    engine = Engine(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        criterion=criterion,
        gradient_handlers=gradient_handlers,
        clip_grad_norm=clip_grad_norm,
    )

    trainer = Trainer(engine, scheduler)

    return trainer, train_dataloader, test_dataloader, lr_scheduler
