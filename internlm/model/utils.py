#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

from internlm.core.context import global_context as gpc


def _split(input_, parallel_mode, dim=-1):
    # skip if only one rank involved
    world_size = gpc.get_world_size(parallel_mode)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = gpc.get_local_rank(parallel_mode)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, parallel_mode, dim=-1):
    # skip if only one rank involved
    world_size = gpc.get_world_size(parallel_mode)
    if world_size == 1:
        return input_

    # all gather
    rank = gpc.get_local_rank(parallel_mode)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    group = gpc.get_cpu_group(parallel_mode) if input_.device.type == "cpu" else gpc.get_group(parallel_mode)
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(input_):
        return _gather(input_, parallel_mode=None)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _gather(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.mode, ctx.dim), None, None


def gather_forward_split_backward(input_, parallel_mode, dim):
    return _GatherForwardSplitBackward.apply(input_, parallel_mode, dim)


def try_import_RMSNorm():
    """
    Try import MixFusedRMSNorm from apex, if failed, return our RMSNorm
    
    """
    try:
        from apex.normalization.fused_layer_norm import MixedFusedRMSNorm as RMSNorm
        return RMSNorm
    except ModuleNotFoundError as e:
        import warnings
        warnings.warn("The torch implementation for MixFusedRMSNorm is slower than apex. Please note this!")
        from internlm.model.norm import RMSNormTorch as RMSNorm
        return RMSNorm