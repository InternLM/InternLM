#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/amp

from functools import partial
from typing import Any, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ReduceOp

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc


def set_fp32_attr_to_module(module: nn.Module):
    setattr(module, "is_fp32_module", True)


def module_has_fp32_attr(module: nn.Module):
    return hasattr(module, "is_fp32_module") and getattr(module, "is_fp32_module")


def set_output_attr_to_module(module: nn.Module):
    setattr(module, "is_output", True)


def module_is_output(module: nn.Module):
    return hasattr(module, "is_output") and getattr(module, "is_output")


class NaiveAMPModel(nn.Module):
    """
    This is a wrapper class for a model that automatically casts the model, its inputs, and outputs into fp16.
    It also provides options to cast the output back to fp32 and to synchronize buffers.

    Args:
        model (torch.nn.Module): The model to be wrapped and cast into fp16.
        output_to_fp32 (bool, optional): If True, the output of this module is cast into fp32. Defaults to True.
        parallel_mode (:class:`internlm.core.context.ParallelMode`): The parallel group mode used in this module.
                                                                Defaults to ``ParallelMode.DATA``.
        sync_buffer (bool, optional): If True, the buffers are synchronized. Defaults to True.
    """

    def __init__(
        self,
        model: nn.Module,
        output_to_fp32: bool = True,
        parallel_mode: ParallelMode = ParallelMode.DATA,
        sync_buffer: bool = True,
        dtype=torch.float16,
    ):
        super().__init__()
        self.model = model.to(dtype)
        self._output_to_fp32 = output_to_fp32
        self._sync_buf = sync_buffer
        self.dtype = dtype

        if gpc.is_initialized(parallel_mode) and gpc.get_world_size(parallel_mode) > 1:
            self._process_group = gpc.get_group(parallel_mode)
            self._world_size = gpc.get_world_size(parallel_mode)
        else:
            self._process_group = None
            self._world_size = 1
            self._sync_buf = False
        self._first_eval_run = False

        # register hook for fp32 module
        self._register_fp32_parameters_hook()

    @property
    def sync_buffer(self):
        """Returns the current state of the buffer synchronization."""
        return self._sync_buf

    @sync_buffer.setter
    def sync_buffer(self, state: bool):
        """Sets the state of the buffer synchronization."""
        self._sync_buf = state

    def _convert_to_fp16(self, input_: Any):
        """Converts the input to fp16 if it is a Tensor of dtype float32."""
        if isinstance(input_, Tensor) and input_.dtype == torch.float32:
            input_ = input_.to(self.dtype)
        return input_

    def _convert_to_fp32(self, input_: Any):
        """Converts the input to fp32 if it is a Tensor of dtype float16."""
        if isinstance(input_, Tensor) and input_.dtype == torch.float16:
            input_ = input_.float()
        return input_

    def convert_to_fp32(self, out):
        """Converts the output to fp32"""
        if isinstance(out, Tensor):
            out = self._convert_to_fp32(out)
        elif isinstance(out, (tuple, list)):
            out = [self._convert_to_fp32(val) for val in out]
        elif isinstance(out, dict):
            out = {key: self._convert_to_fp32(val) for key, val in out.items()}

        return out

    def _reduce_module_buffer(self):
        """
        All-reduces the buffers (e.g., running stats of batch normalization) across
        data parallel ranks so that all the ranks will produce consistent results
        when given the same input.
        """
        buf_list = []

        # find valid buffers
        for buf in self.model.buffers():
            if buf is not None:
                buf_list.append(buf)

        # reduce buffers across data parallel ranks
        if buf_list:
            coalesced_buf = _flatten_dense_tensors(buf_list)
            coalesced_buf.div_(self._world_size)
            dist.all_reduce(coalesced_buf, op=ReduceOp.SUM, group=self._process_group)
            unflattened_buf_list = _unflatten_dense_tensors(coalesced_buf, buf_list)
            for old, new in zip(buf_list, unflattened_buf_list):
                old.copy_(new)

    def eval(self):
        """Sets the model to evaluation mode. Buffers are only synchronized in the first eval iteration."""
        self.model.eval()
        self._first_eval_run = True

    def forward(self, *args, **kwargs):
        """
        Performs a forward pass on the model. Buffers are synchronized before the forward pass.
        The inputs are converted to fp16 and the outputs are optionally converted back to fp32.
        """
        if (self.training or self._first_eval_run) and self._sync_buf:
            with torch.no_grad():
                self._reduce_module_buffer()

            if self._first_eval_run:
                self._first_eval_run = False

        if args:
            args = [self._convert_to_fp16(arg) for arg in args]
        if kwargs:
            for k, v in kwargs.items():
                kwargs[k] = self._convert_to_fp16(v)

        out = self.model(*args, **kwargs)

        if self._output_to_fp32:
            out = self.convert_to_fp32(out)
        return out

    def _register_fp32_parameters_hook(self) -> None:
        """
        Set module to fp32 and register automatic conversion hook in the forward pass.
        The fp32 modules are marked by set_fp32_attr_to_module(.)
        """
        fp32_dtype = torch.float32

        def to_dtype(x, dtype=fp32_dtype):
            if isinstance(x, Tensor) and x.dtype != dtype:
                return x.to(dtype)
            return x

        def _pre_forward_hook_for_fp32(model: nn.Module, inputs: tuple):  # pylint: disable=W0613
            assert isinstance(inputs, tuple)
            return tuple(map(to_dtype, inputs))

        def _post_forward_hook_for_fp32(
            model: nn.Module, inputs: tuple, outputs: Union[tuple, Tensor]
        ):  # pylint: disable=W0613
            assert isinstance(inputs, Union[tuple, Tensor])
            if isinstance(outputs, tuple):
                return tuple(map(to_dtype, outputs, [self.dtype] * len(outputs)))
            else:
                return to_dtype(outputs, self.dtype)

        # just want to share same for loop for ModuleList and Module
        if isinstance(self.model, nn.ModuleList):
            model = self.model
        else:
            model = [self.model]

        modules = []
        # record the modules to transformer/embeding/head/norm block
        for _chunk in model:
            modules.extend([sub_module for _, sub_module in _chunk.named_modules()])

        # register_forward_pre_hook for transformer/embeding/norm/xxx block
        for sub_module in modules:
            if module_has_fp32_attr(sub_module):
                sub_module.to(fp32_dtype)
                sub_module.register_forward_pre_hook(partial(_pre_forward_hook_for_fp32))
                sub_module.register_forward_hook(partial(_post_forward_hook_for_fp32))
            if gpc.config.get("output_tf32", False) and module_is_output(sub_module):
                sub_module.to(fp32_dtype)
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                sub_module.register_forward_pre_hook(partial(_pre_forward_hook_for_fp32))
