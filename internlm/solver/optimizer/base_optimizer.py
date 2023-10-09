#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    """
    Base Optimizer.
    """

    def __init__(self, optim: Optimizer):  # pylint: disable=W0231
        self.optim = optim

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.optim.zero_grad(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.optim.load_state_dict(*args, **kwargs)

    def state_dict(self):
        return self.optim.state_dict()

    def backward(self, loss):
        loss.backward()

    def backward_by_grad(self, tensor, grad):
        torch.autograd.backward(tensors=tensor, grad_tensors=grad)

    def clip_grad_norm(self):
        pass
