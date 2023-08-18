#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.cuda.amp as torch_amp
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class TorchAMPModel(nn.Module):
    """A wrapper class for a model object which executes forward with values automatically
    cast to fp16

    Args:
        model (:class:`torch.nn.Module`): a torch model instance
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @torch_amp.autocast()
    def forward(self, *args, **kwargs):
        """
        Execute forward under the torch amp context
        """
        return self.model(*args, **kwargs)

class TorchAMPLoss(nn.Module):
    """A wrapper class for a criterion object which computes the loss in mixed-precision context

    Args:
        loss (torch.nn.modules.loss._Loss): A loss function object
    """

    def __init__(self, loss: _Loss):
        super().__init__()
        self.loss = loss

    @torch_amp.autocast()
    def forward(self, *args, **kwargs):
        """
        Execute forward under the torch amp context
        """
        return self.loss(*args, **kwargs)

def convert_to_torch_amp(model, criterion):
    
    model = TorchAMPModel(model)
    criterion = TorchAMPLoss(criterion)
    
    return model, criterion