#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn

from .naive_amp import convert_to_naive_amp
from .torch_amp import convert_to_torch_amp


def convert_to_amp(model: nn.Module, criterion, use_amp):
    """A helper function to wrap model with AMP modules.

    Args:
        model (nn.Module): model object.
        criterion: loss function
    Returns:
        model.
    """
    
    if use_amp:
        model, criterion = convert_to_torch_amp(model, criterion)
    else:
        model, criterion = convert_to_naive_amp(model, criterion)
    
    return model, criterion
    
    