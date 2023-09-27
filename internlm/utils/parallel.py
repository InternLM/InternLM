#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.distributed as dist

from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.utils import is_moe_param


def is_model_parallel_parameter(p):
    return hasattr(p, IS_TENSOR_PARALLEL) and getattr(p, IS_TENSOR_PARALLEL)


def sync_tensor(tensor, parallel_mode):
    if gpc.is_initialized(parallel_mode) and gpc.get_world_size(parallel_mode) > 1:
        ranks = gpc.get_ranks_in_group(parallel_mode)
        dist.broadcast(tensor, src=ranks[0], group=gpc.get_group(parallel_mode))


def sync_model_param(model):
    r"""Make sure data parameters are consistent during Data Parallel Mode.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    if gpc.is_initialized(ParallelMode.DATA) and gpc.get_world_size(ParallelMode.DATA) > 1:
        sync_moe_param = (
            gpc.is_initialized(ParallelMode.EXPERT_DATA) and gpc.get_world_size(ParallelMode.EXPERT_DATA) > 1
        )
        for param in model.parameters():
            if sync_moe_param and is_moe_param(param):
                sync_tensor(param, ParallelMode.EXPERT_DATA)
            else:
                sync_tensor(param, ParallelMode.DATA)


def sync_model_param_within_tp(model):
    r"""This function is changed from colossalai, which is ``sync_model_param``.

    We modified this function to make sure it only sync parameters within tensor parallelism
    but they are not splitted by tensor parallelism.
    This function is used to make sure parameters that are not splitted by tensor parallelism
    are the same across each tensor parallelism.
    For example, parameters like RMSNorm, LayerNorm...

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    parallel_mode = ParallelMode.TENSOR
    if gpc.is_initialized(parallel_mode) and gpc.get_world_size(parallel_mode) > 1:
        for param in model.parameters():
            if not is_model_parallel_parameter(param):
                ranks = gpc.get_ranks_in_group(parallel_mode)
                dist.broadcast(param, src=ranks[0], group=gpc.get_group(parallel_mode))


def is_no_pp_or_last_stage():
    return not gpc.is_initialized(ParallelMode.PIPELINE) or gpc.is_last_rank(ParallelMode.PIPELINE)


def get_parallel_log_file_name():
    if gpc.is_rank_for_log():
        fn_prefix = "main_"  # Indicates a rank with more output information
    else:
        fn_prefix = ""

    log_file_name = (
        f"{fn_prefix}dp={gpc.get_local_rank(ParallelMode.DATA)}_"
        f"tp={gpc.get_local_rank(ParallelMode.TENSOR)}_pp={gpc.get_local_rank(ParallelMode.PIPELINE)}"
    )
    return log_file_name
