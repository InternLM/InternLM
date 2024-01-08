from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    """
    All to all communication
    """

    @staticmethod
    def forward(
        ctx: Any,
        # TODO: replace with DS process group
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
        return (None, _AllToAll.apply(ctx.group, *grad_output))
