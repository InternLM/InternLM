#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

from torch import Tensor, nn


def scaled_init_method_normal(sigma: float = 1.0, num_layers: int = 1):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def normal_(mean: float = 0.0, std: float = 1.0):
    r"""Return the initializer filling the input Tensor with values drawn from the normal distribution

     .. math::
        \mathcal{N}(\text{mean}, \text{std}^2)

    Args:
        mean (float): the mean of the normal distribution. Defaults 0.0.
        std (float): the standard deviation of the normal distribution. Defaults 1.0.
    """

    def initializer(tensor: Tensor):
        return nn.init.normal_(tensor, mean, std)

    return initializer


def scaled_init_method_uniform(sigma: float = 1.0, num_layers: int = 1):
    """Init method based on p(x)=Uniform(-a, a) where std(x)=sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    a = math.sqrt(3.0 * std)

    def init_(tensor):
        return nn.init.uniform_(tensor, -a, a)

    return init_


def uniform_(mean: float = 0.0, std: float = 1.0):
    r"""Return the initializer filling the input Tensor with values drawn from the uniform distribution

     .. math::
        \mathcal{U}(mean-a, mean+a), where a satisfies \mathcal{U}_{std}=std.

    Args:
        mean (float): the mean of the uniform distribution. Defaults 0.0.
        std (float): the standard deviation of the uniform distribution. Defaults 1.0.
    """

    a = math.sqrt(3.0 * std)

    def initializer(tensor: Tensor):
        return nn.init.uniform_(tensor, mean - a, mean + a)

    return initializer
