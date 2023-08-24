# adopted from https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm

import numbers

import torch
from torch.nn import init
from torch.nn.parameter import Parameter


def manual_rms_norm(my_input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = my_input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    my_input = my_input * torch.rsqrt(variance + eps)

    if weight is None:
        return my_input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        my_input = my_input.to(weight.dtype)

    return weight * my_input


class RMSNormTorch(torch.nn.Module):
    """A custom PyTorch module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(*normalized_shape))
        self.reset_parameters()

    def forward(self, _input: torch.Tensor):
        return manual_rms_norm(_input, self.normalized_shape, self.weight, self.eps)

    def reset_parameters(self):
        init.ones_(self.weight)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)
