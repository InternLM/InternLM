from typing import TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from internlm.core.context import global_context as gpc
from internlm.moe.experts import Experts

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module


class BaseMoELayer(Base):
    """
    Base MoE Layer.
    """

    def __init__(
        self, gate: Module, experts: Union[Module, ModuleList], ep_group, ep_size: int, num_local_experts: int
    ) -> None:
        super().__init__()
        # for elastic expert paralle, experts may have multiple groups
        expert_group_name = f"moe_ep_size_{ep_size}"
        if expert_group_name not in gpc.expert_parallel_group_names:
            gpc.expert_parallel_group_names.append(expert_group_name)
        self.gate = gate
        self.experts = Experts(experts, num_local_experts, expert_group_name)
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.l_aux = torch.tensor(0.0, device=torch.cuda.current_device(), dtype=gpc.config.model.get("dtype"))
        self.exp_counts = None
