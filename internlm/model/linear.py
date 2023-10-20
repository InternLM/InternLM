#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
from flash_attn.utils.distributed import all_reduce, reduce_scatter
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.utils import (
    Silu,
    fstp_fused_dense_func,
    fused_dense_func_torch,
    megatron_fused_dense_func_torch,
)


class BaseScaleColumnParallelLinear(nn.Linear):
    """
    Base class for ScaleColumnParallelLinear.

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


class ScaleColumnParallelLinear(BaseScaleColumnParallelLinear):
    """
    ScaleColumnParallelLinear in flash implementation.
    """

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


class MegatronScaleColumnParallelLinear(BaseScaleColumnParallelLinear):
    """
    ScaleColumnParallelLinear in megatron implementation.
    """

    def forward(self, input, gather_dim=0):  # pylint: disable=W0622
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight
        return megatron_fused_dense_func_torch(
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
            x,
            self.weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
            gather_dim=gather_dim,
        )


class MegatronColumnParallelLinearTorch(ColumnParallelLinear):
    def forward(self, x, gather_dim=0):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        return megatron_fused_dense_func_torch(
            x,
            self.weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
            gather_dim=gather_dim,
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


class MegatronRowParallelLinearTorch(RowParallelLinear):
    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = megatron_fused_dense_func_torch(x, self.weight, self.bias)
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)


class BaseFeedForward(nn.Module):
    """
    Base FeedForward in flash implementation.

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
        colum_cls=None,
        row_cls=None,
    ):
        super().__init__()
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = colum_cls(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.w2 = colum_cls(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.w3 = row_cls(
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


class FeedForward(BaseFeedForward):
    """
    FeedForward in flash implementation.

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
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            ColumnParallelLinearTorch,
            RowParallelLinearTorch,
        )


class MegatronFeedForward(BaseFeedForward):
    """
    FeedForward in megatron implementation.

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
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            MegatronColumnParallelLinearTorch,
            MegatronRowParallelLinearTorch,
        )


class FSTPLinear(ColumnParallelLinear):
    def forward(self, x):
        block_index = gpc.config.fstp_handler.module_to_index[self]
        return fstp_fused_dense_func(
            x,
            self.weight,
            self.bias,
            process_group=self.process_group,
            module=self,
            handler=gpc.config.fstp_handler,
            block_index=block_index,
            module_name=self._fstp_name,
        )


class FSTPFeedForward(BaseFeedForward):
    """
    FeedForward in FSTP.

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
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            FSTPLinear,
            FSTPLinear,
        )


def get_mlp_cls(sp_mode: str):
    if sp_mode in ["none", "flash-attn"]:
        mlp_cls = FeedForward
    elif sp_mode == "megatron":
        mlp_cls = MegatronFeedForward
    else:
        mlp_cls = FSTPFeedForward
    return mlp_cls


def get_linear_cls(sp_mode: str, parallel_mode: str):
    if parallel_mode == "column":
        if sp_mode in ["none", "flash-attn"]:
            cls = ColumnParallelLinearTorch
        elif sp_mode == "megatron":
            cls = MegatronColumnParallelLinearTorch
        else:
            cls = FSTPLinear
    elif parallel_mode == "row":
        if sp_mode in ["none", "flash-attn"]:
            cls = RowParallelLinearTorch
        elif sp_mode == "megatron":
            cls = MegatronRowParallelLinearTorch
        else:
            cls = FSTPLinear
    return cls
