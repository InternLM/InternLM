#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.cuda.amp as torch_amp
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from internlm.utils.parallel import is_no_pp_or_last_stage


class TorchAMPModel(nn.Module):
    """A wrapper class for a model object which executes forward with values automatically
    cast to fp16

    Args:
        model (:class:`torch.nn.Module`): a torch model instance
    """

    def __init__(self, model: nn.Module, output_to_fp32) -> None:
        super().__init__()
        self.model = model
        self._output_to_fp32 = output_to_fp32

    def _convert_to_fp32(self, input_):
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

    @torch_amp.autocast(dtype=torch.bfloat16)
    def forward(self, *args, **kwargs):
        """
        Execute forward under the torch amp context
        """
        output = self.model(*args, **kwargs)

        with torch.cuda.amp.autocast(enabled=False):
            if self._output_to_fp32:
                output = self.convert_to_fp32(output)
        return output


class TorchAMPLoss(nn.Module):
    """A wrapper class for a criterion object which computes the loss in mixed-precision context

    Args:
        loss (torch.nn.modules.loss._Loss): A loss function object
    """

    def __init__(self, loss: _Loss):
        super().__init__()
        self.loss = loss

    @torch_amp.autocast(dtype=torch.bfloat16)
    def forward(self, *args, **kwargs):
        """
        Execute forward under the torch amp context
        """
        return self.loss(*args, **kwargs)


def convert_to_torch_amp(model, criterion):
    if isinstance(model, nn.ModuleList):
        model = nn.ModuleList(
            [
                TorchAMPModel(
                    model=_m,
                    output_to_fp32=False,  # manually controlled by interleaved pipleline scheduler
                )
                for _m in model
            ]
        )
    else:
        model = TorchAMPModel(
            model=model,
            output_to_fp32=is_no_pp_or_last_stage(),
        )
    criterion = TorchAMPLoss(criterion)

    return model, criterion
