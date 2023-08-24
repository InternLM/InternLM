#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
import torch.nn.functional as F
from flash_attn.ops.fused_dense import FusedDenseFunc
from flash_attn.utils.distributed import (
    all_gather_raw,
    all_reduce_raw,
    reduce_scatter_raw,
)
from torch import Tensor
from torch.cuda.amp import custom_bwd
from torch.distributed import ProcessGroup

from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


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


def linear_bias_wgrad_torch(my_input, grad_output, has_d_bias):
    assert my_input.dtype == grad_output.dtype
    grad_weight = torch.matmul(grad_output.t(), my_input)
    grad_bias = grad_output.sum(dim=0) if has_d_bias else None
    return grad_weight, grad_bias


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py
class FusedDenseFuncTorch(FusedDenseFunc):
    """A custom PyTorch module extending FusedDenseFunc."""

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            else:
                total_x = x
        else:
            (weight,) = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, weight.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]), grad_output, weight)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
                grad_input, handle_grad_input = reduce_fn(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if process_group is not None and sequence_parallel:
                handle_x.wait()
            # we remove the cuda independence, which is different from flash_attn.
            grad_weight, grad_bias = linear_bias_wgrad_torch(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None


def fused_dense_func_torch(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return FusedDenseFunc.apply(x, weight, bias, return_residual, process_group, sequence_parallel)
    else:
        return FusedDenseFuncTorch.apply(x, weight, bias, return_residual, process_group, sequence_parallel)


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(input_):
        return _split(input_, parallel_mode=None)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _split(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.mode, ctx.dim), None, None


def split_forward_gather_backward(input_, parallel_mode, dim):
    return _SplitForwardGatherBackward.apply(input_, parallel_mode, dim)


def try_import_RMSNorm():
    """
    Try import MixFusedRMSNorm from apex, if failed, return our RMSNorm

    """
    try:
        from apex.normalization.fused_layer_norm import MixedFusedRMSNorm as RMSNorm

        return RMSNorm
    except ModuleNotFoundError:
        logger.warning("The torch implementation for MixFusedRMSNorm is slower than apex. Please note this!")
        from internlm.model.norm import RMSNormTorch as RMSNorm

        return RMSNorm
