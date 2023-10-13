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
from internlm.model.utils import (
    Silu,
    all_gather_raw,
    fstp_fused_dense_func,
    fused_dense_func_torch,
)


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
        return fstp_fused_dense_func(
            x, self.weight, self.bias, process_group=self.process_group, module=self, handler=gpc.config.fstp_handler
        )


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
        # import pdb; pdb.set_trace()
        self.process_group = process_group
        self.FSTP_modules = []
        self.module_name = ["Wqkv", "out_proj", "w1", "w2", "w3"]
        self.FSTP_global_weights = dict()  # key: FSTP module; value: module global weight for forward
        self.module_handler = dict()  # key: FSTP module; value: all-gather handler
        self.module_block = dict()  # key: FSTP module; value: transformer block index
        self.block_module = dict()  # key: transformer block index; value: {name_index: FSTP module}
        self.module_name_index = dict()  # key: FSTP module; value: the name in index in self.module_name

        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]

        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model

            for _, children in _chunk.named_children():
                if isinstance(children, nn.ModuleList):
                    for idx, block in enumerate(children):
                        index = 0
                        self.block_module[idx] = {}
                        for _, sub in block.named_children():
                            sub_modules = list(sub.children())
                            if len(sub_modules) > 0:
                                for name, child in sub.named_children():
                                    if isinstance(child, FSTPLinear):
                                        self.FSTP_modules.append(child)
                                        self.module_block[child] = idx
                                        self.block_module[idx][index] = child
                                        self.module_name_index[child] = index
                                        index = index + 1
                            else:
                                continue

    def _register_sync_parameters_hook(self) -> None:
        """
        register pre_forward_hook and pre_backward_hook for FSTPLinear.
        """

        def _pre_forward_hook(module: nn.Module, inputs: Any):
            block_index = self.module_block[module]
            name_index = self.module_name_index[module]
            if name_index == 0:
                total_weight, weight_handler = all_gather_raw(module.weight, self.process_group, async_op=True)
                weight_handler.wait()
                self.FSTP_global_weights[module] = total_weight

                # start the all-gather for next module
                next_module = self.block_module[block_index][name_index + 1]
                self.FSTP_global_weights[next_module], weights_handler = all_gather_raw(
                    next_module.weight, self.process_group, async_op=True
                )
                self.module_handler[next_module] = weights_handler
            else:
                handler = self.module_handler[module]
                handler.wait()
                if name_index != 4:
                    next_module = self.block_module[block_index][name_index + 1]
                    self.FSTP_global_weights[next_module], weights_handler = all_gather_raw(
                        next_module.weight, self.process_group, async_op=True
                    )
                    self.module_handler[next_module] = weights_handler

        def _post_forward_hook(module: nn.Module, input, output):
            del self.FSTP_global_weights[module]
            del self.module_handler[module]

        def _pre_backward_hook(module: nn.Module, grad_input, grad_output):
            block_index = self.module_block[module]
            name_index = self.module_name_index[module]
            if name_index == 4:
                total_weight, weight_handler = all_gather_raw(module.weight, self.process_group, async_op=True)
                weight_handler.wait()
                self.FSTP_global_weights[module] = total_weight

                # start the all-gather for next module
                next_module = self.block_module[block_index][name_index - 1]
                self.FSTP_global_weights[next_module], weights_handler = all_gather_raw(
                    next_module.weight, self.process_group, async_op=True
                )
                self.module_handler[next_module] = weights_handler
            else:
                handler = self.module_handler[module]
                handler.wait()
                if name_index != 0:
                    next_module = self.block_module[block_index][name_index - 1]
                    self.FSTP_global_weights[next_module], weights_handler = all_gather_raw(
                        next_module.weight, self.process_group, async_op=True
                    )
                    self.module_handler[next_module] = weights_handler

        def _post_backward_hook(module, grad_input, grad_output):
            del self.FSTP_global_weights[module]

        for module in self.FSTP_modules:
            # import pdb; pdb.set_trace()
            module.register_forward_pre_hook(_pre_forward_hook)
            module.register_forward_hook(_post_forward_hook)
            # module.register_backward_pre_hook(_pre_backward_hook)
            # module.register_backward_hook(_post_backward_hook)
            module.register_module_full_backward_pre_hook(_pre_backward_hook)


class CoarseGrainedFSTPAllGatherSyncHandler:
    """
    All-gather handler for overlapping the all-gather in adjcent FSTP block.
    """

    def __init__(self, model: Union[nn.Module, nn.ModuleList], process_group) -> None:
        # import pdb; pdb.set_trace()
        self.process_group = process_group
        self.FSTP_blocks = []
        self.FSTP_outs = []
        self.FSTP_wqkvs = []
        self.module_name = ["Wqkv", "out_proj", "w1", "w2", "w3"]
        self.FSTP_global_handle = dict()  # key: FSTP module; value: module global all-gather op handle
        self.FSTP_global_weights = dict()  # key: FSTP module; value: module global weight for forward
        self.block_handles = dict()  # key: transformer block; value: all-gather handles
        self.module_to_index = dict()  # key: FSTP module; value: transformer block index
        self.block_to_index = dict()  # key: transformer block; value: transformer block index
        self.index_to_block = dict()  # key: transformer block index; value: transformer block
        self.index_to_fsdp_modules = dict()  # key: transformer block index; value: fsdp modules

        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]

        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model

            for _, children in _chunk.named_children():
                if isinstance(children, nn.ModuleList):
                    for idx, block in enumerate(children):
                        self.FSTP_blocks.append(block)
                        self.block_to_index[block] = idx
                        self.index_to_block[idx] = block
                        self.index_to_fsdp_modules[idx] = []
                        for _, sub in block.named_children():
                            sub_modules = list(sub.children())
                            if len(sub_modules) > 0:
                                for name, child in sub.named_children():
                                    # print(f"name: {name}", flush=True)
                                    if name == "out_proj":
                                        self.FSTP_outs.append(child)
                                        self.module_to_index[child] = idx
                                    if name == "Wqkv":
                                        self.FSTP_wqkvs.append(child)
                                        self.module_to_index[child] = idx
                                    if isinstance(child, FSTPLinear):
                                        self.index_to_fsdp_modules[idx].append(child)
                            else:
                                continue

    def _all_gather_block_weight(self, block_index: int):
        block = self.index_to_block[block_index]
        fsdp_modules = self.index_to_fsdp_modules[block_index]
        self.block_handles[block] = []
        for module in fsdp_modules:
            total_weight, weight_handle = all_gather_raw(module.weight, self.process_group, async_op=True)
            self.FSTP_global_weights[module] = total_weight
            self.block_handles[block].append(weight_handle)

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
                self._all_gather_block_weight(block_index + 1)

        def _pre_forward_hook_for_block(block: nn.Module, inputs: Any):
            block_index = self.block_to_index[block]
            if block_index == 0:
                # all gather weight for block 0
                fsdp_modules = self.index_to_fsdp_modules[block_index]
                for module in fsdp_modules:
                    total_weight, weight_handle = all_gather_raw(module.weight, self.process_group, async_op=True)
                    weight_handle.wait()
                    self.FSTP_global_weights[module] = total_weight
            else:
                # wait handle for current block
                handles = self.block_handles[block]
                for handle in handles:
                    handle.wait()

        def _post_forward_hook_for_block(block: nn.Module, input, output):
            block_index = self.block_to_index[block]
            fsdp_modules = self.index_to_fsdp_modules[block_index]
            if block in self.block_handles:
                del self.block_handles[block]
            for module in fsdp_modules:
                del self.FSTP_global_weights[module]

        def _pre_backward_hook_for_wqkv(module: nn.Module, grad_output):
            block_index = self.module_to_index[module]
            # start the all-gather for next block
            if block_index - 1 >= 0:
                self._all_gather_block_weight(block_index - 1)

        def _pre_backward_hook_for_block(block: nn.Module, grad_output):
            block_index = self.block_to_index[block]
            if block_index == gpc.config.NUM_LAYER - 1:
                # all gather weight for the last block
                fsdp_modules = self.index_to_fsdp_modules[block_index]
                for module in fsdp_modules:
                    total_weight, weight_handle = all_gather_raw(module.weight, self.process_group, async_op=True)
                    weight_handle.wait()
                    self.FSTP_global_weights[module] = total_weight
            else:
                # wait handle for current block
                handles = self.block_handles[block]
                for handle in handles:
                    handle.wait()

            # start the all-gather for next block
            if block_index - 1 >= 0:
                self._all_gather_block_weight(block_index - 1)

        def _post_backward_hook_for_block(block: nn.Module, grad_input, grad_output):
            block_index = self.block_to_index[block]
            fsdp_modules = self.index_to_fsdp_modules[block_index]
            if block in self.block_handles:
                del self.block_handles[block]
            for module in fsdp_modules:
                del self.FSTP_global_weights[module]

        for block in self.FSTP_blocks:
            block.register_forward_pre_hook(_pre_forward_hook_for_block)
            block.register_forward_hook(_post_forward_hook_for_block)
            block.register_full_backward_pre_hook(_pre_backward_hook_for_block)
            block.register_full_backward_hook(_post_backward_hook_for_block)

        for out_proj in self.FSTP_outs:
            out_proj.register_forward_pre_hook(_pre_forward_hook_for_out_proj)

        # for wqkv in self.FSTP_wqkvs:
        #     wqkv.register_full_backward_pre_hook(_pre_backward_hook_for_wqkv)
