#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn

from .naive_amp import convert_to_naive_amp
from .torch_amp import convert_to_torch_amp


def convert_to_amp(model: nn.Module, criterion, mode: str):
    """A helper function to wrap model with AMP modules.

    Args:
        model (nn.Module): model object.
        mode (str): amp mode(torch or naive)
    Returns:
        model.
    """
    
    if mode == "torch":
        model, criterion = convert_to_torch_amp(model, criterion)
    elif mode == "naive":
        model, criterion = convert_to_naive_amp(model, criterion)
    
    return model, criterion
    
    