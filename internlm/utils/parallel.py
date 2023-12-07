#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.distributed as dist
from torch import nn

from internlm.core.context import IS_SEQUENCE_PARALLEL, IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.model.utils import try_import_RMSNorm
from internlm.solver.pipeline_utils import partition_uniform

RMSNorm = try_import_RMSNorm()


def is_model_parallel_parameter(p):
    return hasattr(p, IS_TENSOR_PARALLEL) and getattr(p, IS_TENSOR_PARALLEL)


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
            if sync_moe_param and getattr(param, "is_expert", False):
                ranks = gpc.get_ranks_in_group(ParallelMode.EXPERT_DATA)
                dist.broadcast(param, src=ranks[0], group=gpc.get_group(ParallelMode.EXPERT_DATA))
            else:
                ranks = gpc.get_ranks_in_group(ParallelMode.DATA)
                dist.broadcast(param, src=ranks[0], group=gpc.get_group(ParallelMode.DATA))


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


def set_model_params_layer_name(model):
    r"""Set the layer name as an attribute of the model parameters.
    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    all_parts = partition_uniform(gpc.config.model.num_layers, pipeline_size, gpc.config.model.num_chunks)
    parts = all_parts[pipeline_rank]

    if not isinstance(model, nn.ModuleList):
        model = [model]

    for chunk_idx, _chunk in enumerate(model):
        if isinstance(_chunk, NaiveAMPModel):
            _chunk = _chunk.model
        chunk_start = parts[chunk_idx][0]
        # Create a unique layer name based on the block's class name and index
        for _, children in _chunk.named_children():
            if isinstance(children, nn.ModuleList):
                for idx, block in enumerate(children):
                    for param_name, param in block.named_parameters():
                        layer_name = f"{block.__class__.__name__}Block{idx + chunk_start}"
                        layer_param_name = f"{layer_name}-{param_name}"
                        param.__setattr__("layer_name", layer_name)
                        param.__setattr__("param_name", layer_param_name)
            else:
                for param_name, param in children.named_parameters():
                    layer_name = f"{children.__class__.__name__}"
                    layer_param_name = f"{layer_name}-{param_name}"
                    param.__setattr__("layer_name", layer_name)
                    param.__setattr__("param_name", f"{layer_name}-{param_name}")


def check_sequence_parallel(model):
    """
    check whether the norm module has IS_SEQUENCE_PARALLEL attribute.
    when the sequence_parallel is True, the norm module should have the IS_SEQUENCE_PARALLEL attribute
    to illustrate the norm should conduct the all-reduce for its grad.
    """

    if not isinstance(model, nn.ModuleList):
        model = [model]

    for _chunk in model:
        if isinstance(_chunk, NaiveAMPModel):
            _chunk = _chunk.model

        for _, module in _chunk.named_modules():
            if isinstance(module, (RMSNorm, nn.LayerNorm)):
                for param in module.parameters():
                    assert hasattr(param, IS_SEQUENCE_PARALLEL), (
                        "when the gpc.config.parallel.sequence parallel is True,"
                        "the params of norm module should have IS_SEQUENCE_PARALLEL attribute"
                    )
