#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from internlm.core.context import global_context as gpc


class BaseGradientHandler(ABC):
    """A basic helper class to handle all-reduce operations of gradients across different parallel groups
    before optimization.

    Args:
        model (Module): Model where the gradients accumulate.
        optimizer (Optimizer): Optimizer for updating the parameters.
    """

    def __init__(self, model, optimizer):
        self._model = model
        self._optimizer = optimizer

    @abstractmethod
    def handle_gradient(self):
        """A method to accumulate gradients across different parallel groups. Users should
        write their own functions or just use the functions in pre-defined subclasses.
        """
        pass


class PipelineSharedModuleGradientHandler(BaseGradientHandler):
    """A helper class to handle all-reduce operations in sub parallel groups.
    A all-reduce collective communication will be operated in
    :func:`handle_gradient` among all sub pipeline parallel groups.
    For better performance, it bucketizes the gradients of all parameters that are
    the same type to improve the efficiency of communication.

    Args:
        model (Module): Model where the gradients accumulate.
        optimizer (Optimizer): Optimizer for updating the parameters.
    """

    def handle_gradient(self):
        """A method running a all-reduce operation in sub pipeline parallel groups."""
        if gpc.pipeline_parallel_size > 1:
            # bucketize and all-reduce
            buckets = defaultdict(lambda: defaultdict(list))
            # Pack the buckets.
            for param in self._model.parameters():
                group = getattr(param, "pipeline_shared_module_pg", None)
                if (
                    param.requires_grad
                    and group is not None
                    and (
                        (hasattr(param, "colo_attr") and not param.colo_attr.saved_grad.is_null())
                        or param.grad is not None
                    )
                ):
                    tp = param.data.type()
                    buckets[group][tp].append(param)

            # For each bucket, all-reduce and copy all-reduced grads.
            for group, group_buckets in buckets.items():
                for tp, bucket in group_buckets.items():
                    grads = [
                        param.colo_attr.grad_payload if hasattr(param, "colo_attr") else param.grad.data
                        for param in bucket
                    ]
                    coalesced = _flatten_dense_tensors(grads).to(torch.cuda.current_device())
                    dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=group)
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)
