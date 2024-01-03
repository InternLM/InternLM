from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module


class BaseMoELayer(Base):
    """
    Base MoE Layer.
    """

    def __init__(self, gate: Module, experts: Module, ep_group, ep_size, num_local_experts: int) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
