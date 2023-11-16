from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc


# Based on https://github.com/pytorch/pytorch/pull/40762
class moe_all_to_all(torch.autograd.Function):
    """
    All to all communication
    """

    @staticmethod
    def forward(
        ctx: Any,
        group: torch.distributed.ProcessGroup,
        inputs: Tensor,
    ) -> Tensor:  # type: ignore
        ctx.group = group
        inputs = inputs.contiguous()
        output = torch.empty_like(inputs)
        dist.all_to_all_single(output, inputs, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, moe_all_to_all.apply(ctx.group, *grad_output))


class moe_stream_acquire(torch.autograd.Function):
    """
    switch to stream
    """

    @staticmethod
    def forward(
        ctx: Any,
        stream,
        event,
    ):
        ctx.origin_stream = torch.cuda.current_stream()
        ctx.event = event
        event.wait(stream)
        torch.cuda.set_stream(stream)

    @staticmethod
    def backward(ctx: Any):
        ctx.event.record(ctx.origin_stream)
        torch.cuda.set_stream(ctx.origin_stream)
        return None, None


class moe_stream_release(torch.autograd.Function):
    """
    switch back to stream
    """

    @staticmethod
    def forward(
        ctx: Any,
        stream,
        event,
    ) -> Tensor:  # type: ignore
        ctx.origin_stream = stream
        ctx.event = event
        event.record(stream)
        torch.cuda.set_stream(torch.cuda.default_stream())

    @staticmethod
    def backward(ctx: Any):
        ctx.event.wait(ctx.origin_stream)
        torch.cuda.set_stream(ctx.origin_stream)
        return None, None


# NOTE: no use due to workload less than 1M
# # Based on https://arxiv.org/pdf/2206.03382.pdf
def _2DHAllToAll(inputs):
    output = torch.empty_like(inputs)
    length = inputs.shape[0]
    slice_size = length // gpc.get_world_size(ParallelMode.EXPERT)
    ngpus = 8  # TODO: should set by user
    nnodes = gpc.get_world_size(ParallelMode.EXPERT) // ngpus

    # phase 0. per-gpu (ngpus) stride copy
    width = nnodes
    height = ngpus
    for i in range(length):
        index = i / slice_size
        offset = i % slice_size
        j = int((width * (index % height) + (index / height)) * slice_size + offset)
        output[j] = inputs[i]
    # print("after intra swap from rank ", gpc.get_global_rank(), " : ", output, flush=True)

    # phase 1. intra-node alltoall
    reqs = []
    node_rank = int(gpc.get_local_rank(ParallelMode.EXPERT) / ngpus)
    for i in range(ngpus):
        reqs.append(
            dist.P2POp(
                dist.isend, output[i * nnodes * slice_size : (i + 1) * nnodes * slice_size], i + node_rank * ngpus
            )
        )
        reqs.append(
            dist.P2POp(
                dist.irecv, inputs[i * nnodes * slice_size : (i + 1) * nnodes * slice_size], i + node_rank * ngpus
            )
        )

    if len(reqs) > 0:
        reqs = dist.batch_isend_irecv(reqs)

    for req in reqs:
        req.wait()
    # print("after intra communication from rank ", gpc.get_global_rank(), " : ", inputs, flush=True)

    # phase 2. per-gpu (nnodes) stride copy
    width = ngpus
    height = nnodes
    for i in range(length):
        index = i / slice_size
        offset = i % slice_size
        j = int((width * (index % height) + (index / height)) * slice_size + offset)
        output[j] = inputs[i]
    # print("after inter swap from rank ", gpc.get_global_rank(), " : ", output, flush=True)

    # phase 3. inter-node alltoall
    reqs = []
    node_rank = int(gpc.get_local_rank(ParallelMode.EXPERT) / ngpus)
    g_local_rank = int(gpc.get_local_rank(ParallelMode.EXPERT) % ngpus)
    for i in range(nnodes):
        reqs.append(
            dist.P2POp(
                dist.isend, output[i * ngpus * slice_size : (i + 1) * ngpus * slice_size], i * ngpus + g_local_rank
            )
        )
        reqs.append(
            dist.P2POp(
                dist.irecv, inputs[i * ngpus * slice_size : (i + 1) * ngpus * slice_size], i * ngpus + g_local_rank
            )
        )

    if len(reqs) > 0:
        reqs = dist.batch_isend_irecv(reqs)

    for req in reqs:
        req.wait()
    # print("after inter communication from rank ", gpc.get_global_rank(), " : ", inputs, flush=True)

    return inputs
