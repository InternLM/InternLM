#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import ProcessGroup

from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.utils import all_reduce, fused_dense_func_torch, reduce_scatter


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
        sequence_parallel: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by " f"world_size ({world_size})")
        super().__init__(in_features, out_features // world_size, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.weight_scale = weight_scale

    def forward(self, input):  # pylint: disable=W0622
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight
        return fused_dense_func_torch(
            input, weight, self.bias, process_group=self.process_group, sequence_parallel=self.sequence_parallel
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
        sequence_parallel: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
    ) -> None:
        super().__init__(in_features, out_features, process_group, bias, sequence_parallel, device, dtype, weight_scale)
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
            input, weight, self.bias, process_group=self.process_group, sequence_parallel=self.sequence_parallel
        )


class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by " f"world_size ({world_size})")
        super().__init__(in_features, out_features // world_size, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.

        return fused_dense_func_torch(
            x, self.weight, self.bias, process_group=self.process_group, sequence_parallel=self.sequence_parallel
        )


class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if in_features % world_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by " f"world_size ({world_size})")
        # Only rank 0 will have bias
        super().__init__(in_features // world_size, out_features, bias=bias and rank == 0, device=device, dtype=dtype)
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

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

        self.w1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )
        self.w2 = ColumnParallelLinear(
            in_features, hidden_features, process_group, bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.w3 = RowParallelLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )
        # need to assign tp attribute so that colossalai know it is tensor parallel module

        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["w1", "w2", "w3"]:
                for param in getattr(self, name).parameters():
                    setattr(param, IS_TENSOR_PARALLEL, True)

    def forward(self, x):
        out = self.w3(F.silu(self.w1(x)) * self.w2(x))
        return out
