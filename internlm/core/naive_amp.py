#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/amp

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ReduceOp

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc


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
