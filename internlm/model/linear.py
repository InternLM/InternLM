#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
import torch.nn.functional as F
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
from flash_attn.utils.distributed import all_reduce, reduce_scatter, all_gather_raw, reduce_scatter_raw
from torch import Tensor
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd

# import fused_dense_cuda  # from apex
import fused_dense_lib as fused_dense_cuda

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.utils import Silu, fused_dense_func_torch


class ScaleColumnParallelLinear(nn.Linear):
    """
    ScaleColumnParallelLinear.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        sequence_parallel (bool): If sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
                                    we do an all_gather of x before doing the matmul.
                                    If not, then the input is already gathered.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by " f"world_size ({world_size})")
        super().__init__(in_features, out_features // world_size, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group
        self.weight_scale = weight_scale

    def forward(self, input, gather_dim=0):  # pylint: disable=W0622
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight
        return fused_dense_func_torch(
            input,
            weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            gather_dim=gather_dim,
        )


class RewardModelLinear(ScaleColumnParallelLinear):
    """
    RewardModelLinear.
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        sequence_parallel (bool): If sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
                                    we do an all_gather of x before doing the matmul.
                                    If not, then the input is already gathered.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
    ) -> None:
        super().__init__(in_features, out_features, process_group, bias, device, dtype, weight_scale)
        torch.distributed.broadcast(self.weight, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], process_group)
        if bias:
            torch.distributed.broadcast(self.bias, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], process_group)

    def forward(self, input):  # pylint: disable=W0622
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight
        return fused_dense_func_torch(
            input,
            weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
        )


class ColumnParallelLinearTorch(ColumnParallelLinear):
    def forward(self, x, gather_dim=0):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.

        return fused_dense_func_torch(
            x, self.weight, self.bias, process_group=self.process_group, sequence_parallel=self.sequence_parallel, gather_dim=gather_dim,
        )


class RowParallelLinearTorch(RowParallelLinear):
    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = fused_dense_func_torch(x, self.weight, self.bias)
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)


class FeedForward(nn.Module):
    """
    FeedForward.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__()

        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinearTorch(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.w2 = ColumnParallelLinearTorch(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.w3 = RowParallelLinearTorch(
            hidden_features,
            out_features,
            process_group,
            bias=bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        w1_o = self.w1(x)
        w2_o = self.w2(x)
        out = self.w3(Silu(w1_o, w2_o))
        return out

class FSDPFusedDenseFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, return_residual=False, process_group=None):

        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        total_x = x.contiguous()
        
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        if world_size > 1:
            # do all_gather for weight and bias before actual computation
            total_weight, handle_weight = all_gather_raw(weight, process_group, async_op=True)
            if bias is not None:
                total_bias, handle_bias = all_gather_raw(bias, process_group, async_op=True)
                handle_bias.wait()
            else:
                total_bias = bias
            handle_weight.wait()
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
            raise RuntimeError('fused_dense only supports matrix dims <= 2M')
        output = F.linear(total_x, total_weight, total_bias)
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
            grad_input, = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            total_x = x
        else:
            weight, = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        if world_size > 1:
            # do all-gather for weight before backward
            total_weight, handle_weight = all_gather_raw(weight, process_group, async_op=True)
            handle_weight.wait()
        else:
            total_weight = weight
        
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, total_weight.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]),
                                         grad_output, total_weight)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
        else:
            grad_input = None

        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient

            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
            if world_size > 1:
                grad_weight, handle_grad_weight = reduce_scatter_raw(grad_weight, process_group, async_op=True)
                if grad_bias is not None:
                    grad_bias, handle_grad_bias = reduce_scatter_raw(grad_bias, process_group, async_op=True) 
                    handle_grad_bias.wait()
                handle_grad_weight.wait()
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        return grad_input, grad_weight, grad_bias, None, None, None


def fsdp_fused_dense_func(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                     return_residual: bool = False, process_group = None):
    dtype_eligible = (x.dtype in [torch.float16, torch.bfloat16]
                      or (x.dtype == torch.float32 and torch.is_autocast_enabled()))
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return FSDPFusedDenseFunc.apply(x, weight, bias, return_residual, process_group)
    else:
        assert process_group is None
        out = F.linear(x, weight, bias)
        return out if not return_residual else (out, x)

class FSDPLinear(ColumnParallelLinear):
    
    def forward(self, x):
        return fsdp_fused_dense_func(x, self.weight, self.bias, process_group=self.process_group)


class FSDPScaleLinear(ScaleColumnParallelLinear):
    
    def forward(self, input):  # pylint: disable=W0622
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight
        return fsdp_fused_dense_func(
            input,
            weight,
            self.bias,
            process_group=self.process_group,
        )


class FSDPFeedForward(nn.Module):
    """
    FeedForward.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__()

        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = FSDPLinear(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.w2 = FSDPLinear(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.w3 = FSDPLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        w1_o = self.w1(x)
        w2_o = self.w2(x)
        out = self.w3(F.silu(w1_o) * w2_o)
        return out
