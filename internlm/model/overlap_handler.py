#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Any, Union

import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.scheduler import SchedulerHook
from internlm.model.embedding import Embedding1D
from internlm.model.linear import FSTPLinear, ScaleColumnParallelLinear
from internlm.model.utils import (
    all_gather_raw_bias_memory_pool,
    all_gather_raw_memory_pool,
)
from internlm.utils.common import get_current_device


class FSTPOverlapHandler:
    """
    FSTP overlap handler for managing the all-gather and reduce_scatter overlapping.
    """

    def __init__(self, model: Union[nn.Module, nn.ModuleList], process_group) -> None:
        self.process_group = process_group
        self.fstp_outs = []
        self.fstp_modules = []
        self.module_name = ["Wqkv", "out_proj", "w1", "w2", "w3"]
        self.fstp_global_handle = dict()  # key: fstp module; value: module global all-gather op handle
        self.bias_global_handle = dict()  # key: fstp module; value: module bias global all-gather op handle
        self.module_to_index = dict()  # key: fstp module; value: transformer block index
        self.index_to_fstp_modules = dict()  # key: transformer block index; value: fsdp modules
        self.last_block = None
        self.head = []
        self.embedding = []
        self.model_checkpoint = gpc.config.model.checkpoint
        self.is_forward = True

        self.reduce_scatter_handlers = {}
        self.zero_const_pool = {}

        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]

        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model

            for _chunk_name, children in _chunk.named_children():
                if isinstance(children, ScaleColumnParallelLinear):
                    self.head.append(children)
                elif isinstance(children, Embedding1D):
                    self.embedding.append(children)
                elif isinstance(children, nn.ModuleList):
                    self.last_block = children[len(children) - 1]
                    for idx, block in enumerate(children):
                        self.index_to_fstp_modules[idx] = []
                        for _sub_name, sub in block.named_children():
                            sub_modules = list(sub.children())
                            if len(sub_modules) > 0:
                                for name, child in sub.named_children():
                                    if name == "out_proj":
                                        self.fstp_outs.append(child)
                                        self.module_to_index[child] = idx
                                    if isinstance(child, FSTPLinear):
                                        self.module_to_index[child] = idx
                                        self.fstp_modules.append(child)
                                        self.index_to_fstp_modules[idx].append(child)

                                        setattr(child, "_fstp_name", name)

                                        if gpc.config.parallel["tensor"].get("reduce_scatter_overlap", False):
                                            _full_name = f"{_chunk_name}.{idx}.{_sub_name}.{name}"
                                            setattr(child.weight, "_fstp_reduce_scatter_str", f"{_full_name}.weight")
                                            if child.bias is not None:
                                                setattr(child.bias, "_fstp_reduce_scatter_str", f"{_full_name}.bias")

        self.num_blocks = len(self.index_to_fstp_modules)

        self._initialize_memory_pool()
        self._register_sync_parameters_hook()

    def get_zero_by_shape(self, size: tuple, dtype, device) -> torch.Tensor:
        if size not in self.zero_const_pool:
            self.zero_const_pool[size] = torch.zeros(*size, dtype=dtype, device=device).contiguous()

        return self.zero_const_pool[size]

    def set_forward_mode(self, flag):
        self.is_forward = flag

    def _initialize_module_shape(self):
        hidden_size = gpc.config.HIDDEN_SIZE
        mlp_ratio = gpc.config.MLP_RATIO
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        mlp_hidden_size = 256 * ((mlp_hidden_size + 256 - 1) // 256)

        self.module_shape["Wqkv"] = (3 * hidden_size, hidden_size)
        self.module_shape["out_proj"] = (hidden_size, hidden_size)
        self.module_shape["w1"] = (mlp_hidden_size, hidden_size)
        self.module_shape["w2"] = (mlp_hidden_size, hidden_size)
        self.module_shape["w3"] = (hidden_size, mlp_hidden_size)

    def _initialize_memory_pool(self) -> None:
        # allocate memory pool
        self.all_gather_memory_pool = []
        self.all_gather_bias_memory_pool = []
        self.reduce_scatter_memory_pool = {}
        self.module_shape = {}

        self._initialize_module_shape()
        dtype = gpc.config.model.get("dtype", torch.half)
        device = get_current_device()

        for _ in range(2):
            weight = {}
            for name in self.module_name:
                weight[name] = torch.zeros(self.module_shape[name], dtype=dtype, device=device).contiguous()
            self.all_gather_memory_pool.append(weight)  # containing two groups of block weight

    def clear_memory_pool(self) -> None:
        self.zero_const_pool = {}
        self.reduce_scatter_memory_pool = {}

    def get_all_gather_memory(self, module):
        block_index = self.module_to_index[module]
        return self.all_gather_memory_pool[block_index % 2][module._fstp_name]

    def get_bias_memory(self, module: nn.Module):
        block_index = self.module_to_index[module]
        # if the bias memory pool is empty or module has been not allocated memory
        if len(self.all_gather_bias_memory_pool) == 0:
            for _ in range(2):
                weight = {}
                weight[module._fstp_name] = torch.zeros(
                    self.module_shape[module._fstp_name][0],
                    dtype=gpc.config.model.get("dtype", torch.half),
                    device=get_current_device(),
                ).contiguous()
                self.all_gather_bias_memory_pool.append(weight)
        elif module._fstp_name not in self.all_gather_bias_memory_pool[0]:
            for i in range(2):
                self.all_gather_bias_memory_pool[i][module._fstp_name] = torch.zeros(
                    self.module_shape[module._fstp_name][0],
                    dtype=gpc.config.model.get("dtype", torch.half),
                    device=get_current_device(),
                ).contiguous()

        return self.all_gather_bias_memory_pool[block_index % 2][module._fstp_name]

    def get_reduce_scatter_memory(self, key):
        # if key not in dict
        if key not in self.reduce_scatter_memory_pool:
            self.reduce_scatter_memory_pool[key] = []

        for index, mem_item in enumerate(self.reduce_scatter_memory_pool[key]):
            if mem_item.idle is True:
                self.reduce_scatter_memory_pool[key][index].idle = False
                return self.reduce_scatter_memory_pool[key][index]

        # if the memory pool is all used
        cur_len = len(self.reduce_scatter_memory_pool[key])
        self.reduce_scatter_memory_pool[key].append(
            torch.zeros(key, dtype=gpc.config.model.get("dtype", torch.half), device=get_current_device()).contiguous()
        )
        setattr(self.reduce_scatter_memory_pool[key][cur_len], "idle", False)
        setattr(self.reduce_scatter_memory_pool[key][cur_len], "index", cur_len)
        return self.reduce_scatter_memory_pool[key][cur_len]

    def release_reduce_scatter_memory(self, key, index):
        self.reduce_scatter_memory_pool[key][index].idle = True

    def _all_gather_block_weight_memory_pool(self, block_index: int):
        fstp_modules = self.index_to_fstp_modules[block_index]
        for module in fstp_modules:
            if module.bias is not None:
                bias_handle = all_gather_raw_bias_memory_pool(
                    module.bias,
                    self.process_group,
                    async_op=True,
                    module=module,
                )
                self.bias_global_handle[module] = bias_handle

            weight_handle = all_gather_raw_memory_pool(
                module.weight,
                self.process_group,
                async_op=True,
                module=module,
            )
            self.fstp_global_handle[module] = weight_handle

    def _register_sync_parameters_hook(self) -> None:
        """
        register forward hooks and backward hooks for fstp modules.
        """

        def _post_forward_hook_for_embedding(module: nn.Module, inputs: Any, output: Any):  # pylint: disable=W0613
            self._all_gather_block_weight_memory_pool(0)

        def _pre_forward_hook_for_out_proj(module: nn.Module, inputs: Any):  # pylint: disable=W0613
            block_index = self.module_to_index[module]
            if self.model_checkpoint and self.is_forward is False:
                if block_index - 1 >= 0:
                    self._all_gather_block_weight_memory_pool(block_index - 1)
            else:
                # start the all-gather for next block
                if block_index + 1 < self.num_blocks:
                    self._all_gather_block_weight_memory_pool(block_index + 1)

        def _pre_forward_hook_for_module(module: nn.Module, inputs: Any):  # pylint: disable=W0613
            if module in self.fstp_global_handle:
                handle = self.fstp_global_handle[module]
                handle.wait()
                if module.bias is not None:
                    bias_handle = self.bias_global_handle[module]
                    bias_handle.wait()
            else:
                weight_handle = all_gather_raw_memory_pool(
                    module.weight,
                    self.process_group,
                    async_op=True,
                    module=module,
                )
                self.fstp_global_handle[module] = weight_handle
                weight_handle.wait()

        def _pre_forward_hook_for_block(block: nn.Module, inputs: Any):  # pylint: disable=W0613
            fstp_modules = self.index_to_fstp_modules[self.num_blocks - 1]
            if module in fstp_modules:
                weight_handle = all_gather_raw_memory_pool(
                    module.weight,
                    self.process_group,
                    async_op=True,
                    module=module,
                )
                self.fstp_global_handle[module] = weight_handle
                weight_handle.wait()

        def _post_forward_hook_for_module(module: nn.Module, inputs: Any, output: Any):  # pylint: disable=W0613
            if module in self.fstp_global_handle:
                del self.fstp_global_handle[module]

        def _post_backward_hook_for_head(module: nn.Module, grad_input, grad_output):  # pylint: disable=W0613
            first_backward_module = self.fstp_modules[-1]
            weight_handle = all_gather_raw_memory_pool(
                first_backward_module.weight,
                self.process_group,
                async_op=True,
                module=first_backward_module,
            )
            self.fstp_global_handle[first_backward_module] = weight_handle

        def _pre_backward_hook_for_head(module: nn.Module, grad_output):
            if self.is_forward is False:
                self._all_gather_block_weight_memory_pool(self.num_blocks - 1)

        def _pre_backward_hook_for_module(module: nn.Module, grad_output):  # pylint: disable=W0613
            # wait handle for current module
            if module in self.fstp_global_handle:
                weight_handle = self.fstp_global_handle[module]
                weight_handle.wait()
            else:
                weight_handle = all_gather_raw_memory_pool(
                    module.weight,
                    self.process_group,
                    async_op=True,
                    module=module,
                )
                self.fstp_global_handle[module] = weight_handle
                weight_handle.wait()

            # start the all-gather for next module
            module_index = self.fstp_modules.index(module)
            if module_index - 1 >= 0:
                next_module = self.fstp_modules[module_index - 1]
                weight_handle = all_gather_raw_memory_pool(
                    next_module.weight,
                    self.process_group,
                    async_op=True,
                    module=next_module,
                )
                self.fstp_global_handle[next_module] = weight_handle

        def _post_backward_hook_for_module(module, grad_input, grad_output):  # pylint: disable=W0613
            if module in self.fstp_global_handle:
                del self.fstp_global_handle[module]

        # register forward hooks
        # 1. register post_forward_hook @embedding module to prefetch for block 0
        # 2. register pre_forward_hook @out_proj module to prefetch for next block,
        #    notice that next block's all_gather op should be after current block's all_to_all op
        # 3. register pre_forward_hook @fstp_module to wait handle for current module
        # 4. register post_forward_hook @fstp_module to release resource
        for embedding in self.embedding:
            embedding.register_forward_hook(_post_forward_hook_for_embedding)

        if self.model_checkpoint:
            if gpc.is_last_rank(parallel_mode=ParallelMode.PIPELINE):
                for head in self.head:
                    head.register_full_backward_pre_hook(_pre_backward_hook_for_head)
            else:
                self.last_block.register_forward_pre_hook(_pre_forward_hook_for_block)

        for out_proj in self.fstp_outs:
            out_proj.register_forward_pre_hook(_pre_forward_hook_for_out_proj)

        for module in self.fstp_modules:
            module.register_forward_pre_hook(_pre_forward_hook_for_module)
            module.register_forward_hook(_post_forward_hook_for_module)

        # register backward hooks
        # 1. register post_backward_hook @head module to prefetch for the last block's last module
        # 2. register pre_backward_hook @fstp_module to wait handle for current module and to prefetch for next module
        # 3. register post_backward_hook @fstp_module to release resource
        if not self.model_checkpoint:
            for head in self.head:
                head.register_full_backward_hook(_post_backward_hook_for_head)

            for module in self.fstp_modules:
                module.register_full_backward_pre_hook(_pre_backward_hook_for_module)
                module.register_full_backward_hook(_post_backward_hook_for_module)


class FSTPOverlapSchedulerHook(SchedulerHook):
    """
    SchedulerHook for fstp overlap handler
    """

    def __init__(self, overlap_handler: FSTPOverlapHandler, zero_optim) -> None:
        self._overlap_handler = overlap_handler
        self._zero_optim = zero_optim

    def before_forward(self, scheduler, inputs) -> None:
        if self._overlap_handler.model_checkpoint:
            self._overlap_handler.set_forward_mode(True)

    def after_forward(self, scheduler, outputs) -> None:
        pass

    def before_criterion(self, scheduler, outputs, label) -> None:
        pass

    def after_criterion(self, scheduler, loss) -> None:
        pass

    def before_backward(self, scheduler, outputs, outputs_grad) -> None:
        if self._overlap_handler.model_checkpoint:
            self._overlap_handler.set_forward_mode(False)

    def after_backward(self, scheduler, inputs_grad) -> None:
        self._zero_optim.accumulate_left_grads_after_backward()

    def post_helper_func(self, scheduler, outputs, label) -> None:
        pass
