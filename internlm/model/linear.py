#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
from flash_attn.utils.distributed import all_reduce, reduce_scatter
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.model.embedding import Embedding1D
from internlm.model.utils import (
    Silu,
    all_gather_raw,
    all_gather_raw_memory_pool,
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
        colum_cls = None,
        row_cls = None,
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
        super().__init__(in_features, hidden_features, out_features, process_group, bias, device, 
                         dtype, multiple_of, ColumnParallelLinearTorch, RowParallelLinearTorch)
       

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
        super().__init__(in_features, hidden_features, out_features, process_group, bias, device,
                         dtype, multiple_of, MegatronColumnParallelLinearTorch, MegatronRowParallelLinearTorch)

class FSTPLinear(ColumnParallelLinear):
    def forward(self, x):
        block_index = gpc.config.fstp_handler.module_to_index[self]
        name_index = gpc.config.fstp_handler.module_name_index[self]
        name = gpc.config.fstp_handler.module_name[name_index]
        return fstp_fused_dense_func(
            x, self.weight, self.bias, process_group=self.process_group, 
            module=self, handler=gpc.config.fstp_handler, block_index=block_index, module_name=name
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
        super().__init__(in_features, hidden_features, out_features, process_group, bias, device,
                         dtype, multiple_of, FSTPLinear, FSTPLinear)

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
    elif parallel_mode == 'row':
        if sp_mode in ["none", "flash-attn"]:
            cls = RowParallelLinearTorch
        elif sp_mode == "megatron":
            cls = MegatronRowParallelLinearTorch
        else:
            cls = FSTPLinear
    return cls

class CoarseGrainedFSTPAllGatherSyncHandler:
    """
    All-gather handler for overlapping the all-gather in adjcent FSTP block.
    """

    def __init__(self, model: Union[nn.Module, nn.ModuleList], process_group) -> None:
        # import pdb; pdb.set_trace()
        self.process_group = process_group
        self.FSTP_blocks = []
        self.FSTP_outs = []
        self.FSTP_modules = []
        self.module_name = ["Wqkv", "out_proj", "w1", "w2", "w3"]
        self.FSTP_global_handle = dict()  # key: FSTP module; value: module global all-gather op handle
        self.FSTP_global_weights = dict()  # key: FSTP module; value: module global weight for forward
        self.block_handles = dict()  # key: transformer block; value: all-gather handles
        self.module_to_index = dict()  # key: FSTP module; value: transformer block index
        self.block_to_index = dict()  # key: transformer block; value: transformer block index
        self.index_to_block = dict()  # key: transformer block index; value: transformer block
        self.index_to_fsdp_modules = dict()  # key: transformer block index; value: fsdp modules
        self.module_name_index = dict()  # key: FSTP module; value: the name in index in self.module_name
        self.block_module = dict()  # key: transformer block index; value: {name_index: FSTP module}
        self.head = []
        self.embedding = []

        self.reduce_scatter_handlers = {}
        self.all_reduce_handlers = {}
        self.zero_const_pool = {}

        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]

        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model

            for _chunk_name, children in _chunk.named_children():
                if isinstance(children, nn.ModuleList):
                    for idx, block in enumerate(children):
                        index = 0
                        self.block_module[idx] = {}
                        self.FSTP_blocks.append(block)
                        self.block_to_index[block] = idx
                        self.index_to_block[idx] = block
                        self.index_to_fsdp_modules[idx] = []
                        for _sub_name, sub in block.named_children():
                            sub_modules = list(sub.children())
                            if len(sub_modules) > 0:
                                for name, child in sub.named_children():
                                    if name == "out_proj":
                                        self.FSTP_outs.append(child)
                                        self.module_to_index[child] = idx
                                    if isinstance(child, FSTPLinear):
                                        self.module_to_index[child] = idx
                                        self.block_module[idx][index] = child
                                        self.FSTP_modules.append(child)
                                        self.index_to_fsdp_modules[idx].append(child)
                                        self.module_name_index[child] = index
                                        index = index + 1

                                        _full_name = f"{_chunk_name}.{idx}.{_sub_name}.{name}"
                                        setattr(child.weight, "_fstp_reduce_scatter_str", f"{_full_name}.weight")
                                        if child.bias is not None:
                                            setattr(child.bias, "_fstp_reduce_scatter_str", f"{_full_name}.bias")
                            else:
                                continue
                elif isinstance(children, ScaleColumnParallelLinear):
                    self.head.append(children)
                elif isinstance(children, Embedding1D):
                    self.embedding.append(children)

    def get_zero_by_shape(self, size: tuple, dtype, device) -> torch.Tensor:
        if size not in self.zero_const_pool:
            self.zero_const_pool[size] = torch.zeros(*size, dtype=dtype, device=device).contiguous()

        return self.zero_const_pool[size]

    def _all_gather_block_weight_memory_pool(self, block_index: int):
        fsdp_modules = self.index_to_fsdp_modules[block_index]
        for module in fsdp_modules:
            module_index = self.module_name_index[module]
            name = self.module_name[module_index]
            weight_handle = all_gather_raw_memory_pool(
                module.weight, self.process_group, async_op=True, block_index=block_index, module_name=name
            )
            self.FSTP_global_handle[module] = weight_handle

    def _register_sync_parameters_hook(self) -> None:
        """
        register pre_forward_hook and pre_backward_hook for FSTP block.

        Notice that next block's all_gather op should be after current block's all_to_all op, so we
        1. register pre_forward_hook @out_proj module to prefetch for next block
        2. register pre_forward_hook @block module to wait handles for next block
        3. register pre_backward_hook @wqkv module to prefetch for next block
        4. register pre_backward_hook @block module to wait handles for next block
        """

        def _pre_forward_hook_for_out_proj(module: nn.Module, inputs: Any):
            block_index = self.module_to_index[module]
            # start the all-gather for next block
            if block_index + 1 < gpc.config.NUM_LAYER:
                self._all_gather_block_weight_memory_pool(block_index + 1)

        def _post_forward_hook_for_embedding(module: nn.Module, inputs: Any, output):
            self._all_gather_block_weight_memory_pool(0)

        def _pre_forward_hook_for_module(module: nn.Module, inputs: Any):
            handle = self.FSTP_global_handle[module]
            handle.wait()

        def _post_forward_hook_for_module(module: nn.Module, input, output):
            if module in self.FSTP_global_weights:
                del self.FSTP_global_weights[module]
            if module in self.FSTP_global_handle:
                del self.FSTP_global_handle[module]

        def _post_backward_hook_for_head(module: nn.Module, grad_input, grad_output):
            first_module = self.block_module[gpc.config.NUM_LAYER - 1][4]
            total_weight, weight_handler = all_gather_raw(first_module.weight, self.process_group, async_op=True)
            self.FSTP_global_handle[first_module] = weight_handler
            self.FSTP_global_weights[first_module] = total_weight

        def _pre_backward_hook_for_module_memory_pool(module: nn.Module, grad_output):
            block_index = self.module_to_index[module]
            name_index = self.module_name_index[module]

            if name_index == 4 and block_index == gpc.config.NUM_LAYER - 1:
                weight_handler = self.FSTP_global_handle[module]
                weight_handler.wait()

                # start the all-gather for next module
                next_module = self.block_module[block_index][name_index - 1]
                next_name = self.module_name[name_index - 1]
                weights_handler = all_gather_raw_memory_pool(
                    next_module.weight,
                    self.process_group,
                    async_op=True,
                    block_index=block_index,
                    module_name=next_name,
                )
                self.FSTP_global_handle[next_module] = weights_handler
            elif name_index == 0:
                handler = self.FSTP_global_handle[module]
                handler.wait()

                if block_index - 1 >= 0:
                    next_module = self.block_module[block_index - 1][4]
                    name = self.module_name[4]
                    weights_handler = all_gather_raw_memory_pool(
                        next_module.weight,
                        self.process_group,
                        async_op=True,
                        block_index=block_index - 1,
                        module_name=name,
                    )
                    self.FSTP_global_handle[next_module] = weights_handler
            else:
                handler = self.FSTP_global_handle[module]
                handler.wait()
                if name_index != 0:
                    next_module = self.block_module[block_index][name_index - 1]
                    name = self.module_name[name_index - 1]
                    weights_handler = all_gather_raw_memory_pool(
                        next_module.weight, self.process_group, async_op=True, block_index=block_index, module_name=name
                    )
                    self.FSTP_global_handle[next_module] = weights_handler

        def _post_backward_hook_for_module(module, grad_input, grad_output):
            if module in self.FSTP_global_weights:
                del self.FSTP_global_weights[module]
            if module in self.FSTP_global_handle:
                del self.FSTP_global_handle[module]

        for embedding in self.embedding:
            embedding.register_forward_hook(_post_forward_hook_for_embedding)

        for head in self.head:
            head.register_full_backward_hook(_post_backward_hook_for_head)

        for out_proj in self.FSTP_outs:
            out_proj.register_forward_pre_hook(_pre_forward_hook_for_out_proj)

        for module in self.FSTP_modules:
            module.register_forward_pre_hook(_pre_forward_hook_for_module)
            module.register_forward_hook(_post_forward_hook_for_module)
            module.register_full_backward_pre_hook(_pre_backward_hook_for_module_memory_pool)
            module.register_full_backward_hook(_post_backward_hook_for_module)
