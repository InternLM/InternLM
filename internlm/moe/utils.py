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
        input_splits=None,
        output_splits=None,
    ) -> Tensor:  # type: ignore
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        inputs = inputs.contiguous()
        output = (
            torch.empty_like(inputs)
            if output_splits is None
            else inputs.new_empty(size=[sum(output_splits)] + list(inputs.size()[1:]))
        )
        dist.all_to_all_single(
            output, inputs, output_split_sizes=output_splits, input_split_sizes=input_splits, group=group
        )
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output, ctx.output_splits, ctx.input_splits), None, None)
