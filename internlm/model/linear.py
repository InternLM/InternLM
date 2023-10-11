#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Union, Any

import torch
import torch.nn.functional as F
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
from flash_attn.utils.distributed import all_reduce, reduce_scatter
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.utils import Silu, fstp_fused_dense_func, fused_dense_func_torch


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


class FSTPLinear(ColumnParallelLinear):
    def forward(self, x):
        import pdb; pdb.set_trace()
        return fstp_fused_dense_func(x, self.weight, self.bias, process_group=self.process_group)


class FSTPFeedForward(nn.Module):
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

        self.w1 = FSTPLinear(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.w2 = FSTPLinear(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.w3 = FSTPLinear(
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

class FSTPAllGatherSyncHandler:
    """
    All-gather handler for overlapping the all-gather in adjcent FSTP linear.
    """
    
    def __init__(self, model: Union[nn.Module, nn.ModuleList], process_group) -> None:
        
        self.process_group = process_group
        self.FSTP_modules = []
        self.module_name = ["Wqkv", "out_proj", "w1", "w2", "w3"]
        self.FSTP_global_weights = dict() # key: FSTP module; value: module global weight for forward
        self.module_handler = dict() # key: FSTP module; value: all-gather handler
        self.module_block = dict() # key: FSTP module; value: transformer block index
        self.block_module = dict() # key: transformer block index; value: {name_index: FSTP module}
        self.module_name_index = dict() # key: FSTP module; value: the name in index in self.module_name
        
        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]
        
        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model
            
            for _, children in _chunk.named_children():
                if isinstance(children, nn.ModuleList):
                    for _, block in enumerate(children):
                        index = 0
                        sub_modules = list(block.children())
                        if len(sub_modules) > 0:
                            for name, child in block.named_children():
                                if isinstance(child, FSTPLinear):
                                    self.FSTP_modules.append(child)
                                    self.module_block[child] = _
                                    self.block_module[_][index] = child
                                    self.module_name_index[child] = index
                                    index = index + 1
                        else:
                            continue
        
    
    def _register_sync_parameters_hook(self) -> None:
        """
        register pre_forward_hook and pre_backward_hook for FSTPLinear.
        """
        
        def _hook(module: nn.Module):
            block_index = self.module_block[module]
            name_index = self.module_name_index[module]
            if name_index == 0:
                next_module = self.block_module[block_index][name_index + 1]
                self.FSTP_global_weights, weights_handler = all_gather_raw(next_module.weight, self.process_group, async_op=True)
                self.module_handler[next_module] = weights_handler
            else:
                handler = self.module_handler[module]
                handler.wait()
                if name_index != 4:
                    next_module = self.block_module[block_index][name_index + 1]
                    self.FSTP_global_weights, weights_handler = all_gather_raw(next_module.weight, self.process_group, async_op=True)
                    self.module_handler[next_module] = weights_handler
        
        def _pre_forward_hook(module: nn.Module, inputs: Any):
            _hook(module)

        def _pre_backward_hook(module: nn.Module, grad_input, grad_output):
            _hook(module)
        
        for module in self.FSTP_modules:
            module.register_forward_pre_hook(_pre_forward_hook)
            module.register_backward_pre_hook(_pre_backward_hook)