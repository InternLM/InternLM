#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn.functional as F
from flash_attn.utils.distributed import all_reduce_raw
from torch import Tensor, nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup

from internlm.core.context import ParallelMode
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


def all_gather_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False, gather_dim: int = 0):
    world_size = torch.distributed.get_world_size(process_group)
    shape = list(input_.shape)
    shape[gather_dim] = shape[gather_dim] * world_size
    output = torch.empty(shape, dtype=input_.dtype, device=input_.device)
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


def all_gather_raw_memory_pool(
    input_: Tensor,
    process_group: ProcessGroup,
    async_op: bool = False,
    module: nn.Module = None,
):
    handle = torch.distributed.all_gather_into_tensor(
        gpc.fstp_handler.get_all_gather_memory(module=module),
        input_.contiguous(),
        group=process_group,
        async_op=async_op,
    )
    return handle


def all_gather_raw_bias_memory_pool(
    input_: Tensor,
    process_group: ProcessGroup,
    async_op: bool = False,
    module: nn.Module = None,
):
    handle = torch.distributed.all_gather_into_tensor(
        gpc.fstp_handler.get_bias_memory(module=module),
        input_.contiguous(),
        group=process_group,
        async_op=async_op,
    )
    return handle


def linear_bias_wgrad_torch(my_input, grad_output, has_d_bias):
    assert my_input.dtype == grad_output.dtype
    grad_weight = torch.matmul(grad_output.t(), my_input)
    grad_bias = grad_output.sum(dim=0) if has_d_bias else None
    return grad_weight, grad_bias


def reduce_scatter_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(
        input_.shape[0] // world_size, *input_.shape[1:], dtype=input_.dtype, device=input_.device
    ).contiguous()
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


def reduce_scatter_raw_memory_pool(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    size = (input_.shape[0] // world_size, *input_.shape[1:])
    output = gpc.fstp_handler.get_reduce_scatter_memory(size)
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py
class FusedDenseFunc(torch.autograd.Function):
    "FusedDenseFunc for tensor parallel in flash-attn implementation."

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, return_residual=False, process_group=None, sequence_parallel=True, gather_dim=0):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel
        ctx.gather_dim = gather_dim

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True, gather_dim=gather_dim)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        gather_dim = ctx.gather_dim

        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True, gather_dim=gather_dim)
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
            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


class MegatronFusedDenseFunc(torch.autograd.Function):
    """
    FusedDenseFunc for tensor parallel in megatron implementation.
    The diffenrence between the implementation of flash-attn and megatron is that the total_x could be
    saved for backward in megatron, so that the all-gather in backward is ommited.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, return_residual=False, process_group=None, sequence_parallel=True, gather_dim=0):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True, gather_dim=gather_dim)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(total_x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

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
            total_x, weight = ctx.saved_tensors
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
            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py
class FusedDenseFuncTorch(FusedDenseFunc):
    """FusedDenseFunc in flash implementation for supporting torch.float32"""

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        gather_dim = ctx.gather_dim
        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True, gather_dim=gather_dim)
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
        return grad_input, grad_weight, grad_bias, None, None, None, None


class MegatronFusedDenseFuncTorch(FusedDenseFunc):
    """FusedDenseFunc in megatron implementation for supporting torch.float32"""

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
            total_x, weight = ctx.saved_tensors
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
            # we remove the cuda independence, which is different from flash_attn.
            grad_weight, grad_bias = linear_bias_wgrad_torch(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


class FSTPFusedDenseFunc(torch.autograd.Function):
    "FusedDenseFunc for FSTP, which is optimized based on flash implementation."

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        weight,
        bias,
        return_residual=False,
        process_group=None,
        module=None,
        overlap_handler=None,
    ):
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.overlap_handler = overlap_handler
        ctx.module = module

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        total_x = x.contiguous()

        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        if world_size > 1:
            # do all_gather for weight and bias before actual computation
            if overlap_handler is not None:
                total_weight = gpc.fstp_handler.get_all_gather_memory(module=module)
            else:
                total_weight, handle_weight = all_gather_raw(weight, process_group, async_op=True)
                handle_weight.wait()
            # TODO memory pool for bias
            if bias is not None:
                if overlap_handler is not None:
                    total_bias = gpc.fstp_handler.get_bias_memory(module=module)
                else:
                    total_bias, handle_bias = all_gather_raw(bias, process_group, async_op=True)
                    handle_bias.wait()
            else:
                total_bias = bias
        else:
            total_weight = weight
            total_bias = bias

        if torch.is_autocast_enabled():
            total_weight = total_weight.to(dtype=torch.get_autocast_gpu_dtype())
            total_bias = total_bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None

        total_weight = total_weight.contiguous()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *total_weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        output = F.linear(total_x, total_weight, total_bias)
        # release memory
        del total_weight
        del total_bias
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(weight, bias)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        overlap_handler = ctx.overlap_handler
        module = ctx.module

        if ctx.compute_weight_gradient:
            x, weight, bias = ctx.saved_tensors
            total_x = x
        else:
            weight, bias = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])

        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        if world_size > 1:
            if overlap_handler is not None:
                total_weight = gpc.fstp_handler.get_all_gather_memory(module=module)
            else:
                total_weight, handle_weight = all_gather_raw(weight, process_group, async_op=True)
                handle_weight.wait()
        else:
            total_weight = weight

        # compute weight grad
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
            if world_size > 1:
                if overlap_handler is not None:
                    grad_weight_async, handle_grad_weight = reduce_scatter_raw_memory_pool(
                        grad_weight, process_group, async_op=True
                    )
                    assert hasattr(weight, "_fstp_reduce_scatter_str")
                    overlap_handler.reduce_scatter_handlers[weight._fstp_reduce_scatter_str] = (
                        handle_grad_weight,
                        grad_weight_async,
                    )
                    grad_weight = overlap_handler.get_zero_by_shape(
                        (
                            grad_weight.shape[0] // torch.distributed.get_world_size(process_group),
                            *grad_weight.shape[1:],
                        ),
                        dtype=grad_weight.dtype,
                        device=grad_weight.device,
                    )
                    if grad_bias is not None:
                        grad_bias_async, handle_grad_bias = reduce_scatter_raw_memory_pool(
                            grad_bias, process_group, async_op=True
                        )
                        assert hasattr(bias, "_fstp_reduce_scatter_str")
                        overlap_handler.reduce_scatter_handlers[bias._fstp_reduce_scatter_str] = (
                            handle_grad_bias,
                            grad_bias_async,
                        )
                        grad_bias = overlap_handler.get_zero_by_shape(
                            (
                                grad_bias.shape[0] // torch.distributed.get_world_size(process_group),
                                *grad_bias.shape[1:],
                            ),
                            dtype=grad_bias.dtype,
                            device=grad_bias.device,
                        )
                else:
                    grad_weight, handle_grad_weight = reduce_scatter_raw(grad_weight, process_group, async_op=True)
                    if grad_bias is not None:
                        grad_bias, handle_grad_bias = reduce_scatter_raw(grad_bias, process_group, async_op=True)
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None

        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, total_weight.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]), grad_output, total_weight)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
        else:
            grad_input = None
        del total_weight

        if ctx.needs_input_grad[1]:
            if world_size > 1 and overlap_handler is None:
                handle_grad_weight.wait()
                if grad_bias is not None:
                    handle_grad_bias.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class FSTPFusedDenseFuncTorch(FSTPFusedDenseFunc):
    "FusedDenseFunc for FSTP, which is optimized based on flash implementation."

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        overlap_handler = ctx.overlap_handler
        module = ctx.module

        if ctx.compute_weight_gradient:
            x, weight, bias = ctx.saved_tensors
            total_x = x
        else:
            weight, bias = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])

        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        if world_size > 1:
            if overlap_handler is not None:
                total_weight = gpc.fstp_handler.get_all_gather_memory(module=module)
            else:
                total_weight, handle_weight = all_gather_raw(weight, process_group, async_op=True)
                handle_weight.wait()
        else:
            total_weight = weight

        # compute weight grad
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            grad_weight, grad_bias = linear_bias_wgrad_torch(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
            if world_size > 1:
                if overlap_handler is not None:
                    grad_weight_async, handle_grad_weight = reduce_scatter_raw_memory_pool(
                        grad_weight, process_group, async_op=True
                    )
                    assert hasattr(weight, "_fstp_reduce_scatter_str")
                    overlap_handler.reduce_scatter_handlers[weight._fstp_reduce_scatter_str] = (
                        handle_grad_weight,
                        grad_weight_async,
                    )
                    grad_weight = overlap_handler.get_zero_by_shape(
                        (
                            grad_weight.shape[0] // torch.distributed.get_world_size(process_group),
                            *grad_weight.shape[1:],
                        ),
                        dtype=grad_weight.dtype,
                        device=grad_weight.device,
                    )
                    if grad_bias is not None:
                        grad_bias_async, handle_grad_bias = reduce_scatter_raw_memory_pool(
                            grad_bias, process_group, async_op=True
                        )
                        assert hasattr(bias, "_fstp_reduce_scatter_str")
                        overlap_handler.reduce_scatter_handlers[bias._fstp_reduce_scatter_str] = (
                            handle_grad_bias,
                            grad_bias_async,
                        )
                        grad_bias = overlap_handler.get_zero_by_shape(
                            (
                                grad_bias.shape[0] // torch.distributed.get_world_size(process_group),
                                *grad_bias.shape[1:],
                            ),
                            dtype=grad_bias.dtype,
                            device=grad_bias.device,
                        )
                else:
                    grad_weight, handle_grad_weight = reduce_scatter_raw(grad_weight, process_group, async_op=True)
                    if grad_bias is not None:
                        grad_bias, handle_grad_bias = reduce_scatter_raw(grad_bias, process_group, async_op=True)
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None

        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, total_weight.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]), grad_output, total_weight)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
        else:
            grad_input = None
        del total_weight

        if ctx.needs_input_grad[1]:
            if world_size > 1 and overlap_handler is None:
                handle_grad_weight.wait()
                if grad_bias is not None:
                    handle_grad_bias.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


def fused_dense_func_torch(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
    gather_dim: int = 0,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return FusedDenseFunc.apply(x, weight, bias, return_residual, process_group, sequence_parallel, gather_dim)
    else:
        return FusedDenseFuncTorch.apply(x, weight, bias, return_residual, process_group, sequence_parallel, gather_dim)


def megatron_fused_dense_func_torch(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
    gather_dim: int = 0,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return MegatronFusedDenseFunc.apply(
            x, weight, bias, return_residual, process_group, sequence_parallel, gather_dim
        )
    else:
        return MegatronFusedDenseFuncTorch.apply(
            x, weight, bias, return_residual, process_group, sequence_parallel, gather_dim
        )


def fstp_fused_dense_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group=None,
    module=None,
    handler=None,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return FSTPFusedDenseFunc.apply(x, weight, bias, return_residual, process_group, module, handler)
    else:
        return FSTPFusedDenseFuncTorch.apply(x, weight, bias, return_residual, process_group, module, handler)


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


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "is_expert") and param.is_expert:
        return True
    return False


def is_gate_param(param: torch.Tensor) -> bool:
    if hasattr(param, "is_gate") and param.is_gate:
        return True
    return False


def is_norm_param(param: torch.Tensor) -> bool:
    if hasattr(param, "is_norm") and param.is_norm:
        return True
    return False


def Silu(w1_o, w2_o):
    return F.silu(w1_o) * w2_o


Silu = torch.jit.script(Silu)
