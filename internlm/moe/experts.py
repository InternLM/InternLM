"""
The file has been adapted from the following files:
https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
 Git commit hash: f3943cf9109226ed3ecf2d5dbb639a11cd925555
 We retain the following license from the original files:
"""
from typing import Union, cast

import torch
from torch.nn import Module, ModuleList


class Experts(torch.nn.Module):
    """
    Local Experts.
    """

    def __init__(self, experts: Union[Module, ModuleList], num_local_experts=1, expert_group_name=None):
        super().__init__()

        if isinstance(experts, ModuleList):
            self.wrapped_experts = cast(ModuleList, experts)
        else:
            self.wrapped_experts = ModuleList([experts])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.wrapped_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for _, param in expert.named_parameters():
                param.is_expert = True
                param.group_name = expert_group_name

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.wrapped_experts):
            out = expert(chunk)
            if isinstance(out, tuple):
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output
