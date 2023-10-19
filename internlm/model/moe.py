import typing

import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.linear import FeedForward
from internlm.moe.experts import Experts
from internlm.moe.sharded_moe import MOELayer, TopKGate
from internlm.utils.logger import get_logger

# global llm logger
logger = get_logger(__file__)


class MoE(torch.nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        moe_use_residual (bool, optional): default=False, make this MoE layer a Residual MoE
                                          (https://arxiv.org/abs/2201.05596) layer.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        hidden_size,
        num_experts=1,
        ep_size=1,
        k=1,
        capacity_factor=1.0,
        eval_capacity_factor=1.0,
        min_capacity=4,
        noisy_gate_policy: typing.Optional[str] = None,
        drop_tokens: bool = True,
        use_rts: bool = True,
        using_default_moe: bool = True,
        use_residual=False,
        device=None,
        dtype=None,
    ):

        super().__init__()

        assert (
            num_experts % ep_size == 0
        ), f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        assert noisy_gate_policy is None or noisy_gate_policy in ["None", "Jitter", "RSample"], (
            "Unsupported noisy_gate_policy: " + noisy_gate_policy
        )

        # for elastic expert paralle, experts may have multiple groups
        expert_group_name = f"moe_ep_size_{self.ep_size}"
        if expert_group_name not in gpc.expert_parallel_group_names:
            gpc.expert_parallel_group_names.append(expert_group_name)
        experts = torch.nn.ModuleList(
            [
                FeedForward(
                    hidden_size,
                    int(hidden_size * gpc.config.model.mlp_ratio),
                    out_features=hidden_size,
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_local_experts)
            ]
        )
        experts = Experts(experts, self.num_local_experts, expert_group_name)

        if using_default_moe:
            self.moe_layer = MOELayer(
                TopKGate(
                    hidden_size,
                    num_experts,
                    k,
                    capacity_factor,
                    eval_capacity_factor,
                    min_capacity,
                    noisy_gate_policy,
                    drop_tokens,
                    use_rts,
                ),
                experts,
                gpc.get_group(ParallelMode.EXPERT),
                self.ep_size,
                self.num_local_experts,
            )

        # residual network, see https://arxiv.org/pdf/2201.05596.pdf, seems useful for convergence
        self.use_residual = use_residual
        if use_residual:
            self.residual_mlp = FeedForward(
                hidden_size,
                int(hidden_size * gpc.config.model.mlp_ratio),
                out_features=hidden_size,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias=False,
                device=device,
                dtype=dtype,
            )
            # coefficient is used for weighted sum of the output of expert and residual mlp
            self.coefficient = torch.nn.Linear(hidden_size, 2)

    def forward(self, hidden_states, used_token=None):
        """MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.moe_layer(hidden_states, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.residual_mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.moe_layer.l_aux, self.moe_layer.exp_counts
