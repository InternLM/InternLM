import typing
from typing import Dict, Tuple

import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.moe.experts import Experts
from internlm.moe.sharded_moe import MOELayer, TopKGate
from internlm.utils.logger import get_logger

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


# global llm logger
logger = get_logger(__file__)


def has_moe_layers(m):
    has_moe = False
    num_experts = 0

    for _, module in m.named_modules():
        if isinstance(module, MoE):
            has_moe = True
            num_experts = module.num_experts
            break
    return has_moe, num_experts


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "all_reduce") and not param.all_reduce:
        return True
    return False


class MoE(torch.nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
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
        experts,
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
        use_residual=True,
        residual_mlp=None,
    ):

        super().__init__()

        assert (
            num_experts % ep_size == 0
        ), f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        logger.info(  # pylint: disable=W1203
            f"Creating MoE layer with num_experts: {num_experts} | num_local_experts:"
            f"{self.num_local_experts} | expert_parallel_size: {self.ep_size}"
        )

        assert noisy_gate_policy is None or noisy_gate_policy in ["None", "Jitter", "RSample"], (
            "Unsupported noisy_gate_policy: " + noisy_gate_policy
        )

        expert_group_name = f"ep_size_{self.ep_size}"
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

        self.use_residual = use_residual
        if use_residual:
            self.residual_mlp = residual_mlp
            # coefficient is used for weighted sum of the output of expert and mlp
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


def split_params_into_different_moe_groups_for_optimizer(
    param_groups: Tuple[Dict], max_group_size=178956971
) -> Tuple[Dict]:
    """Split parameters into different MoE groups for optimizer
    Compatiable with muiltiple param groups, each should have a name

    Args:
        param_groups (Tuple[Dict]):
            The list of parameter groups to split

    Returns:
        Tuple[Dict]:
        list of MoE/non-MoE groups for optimizer
    """
    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    # gather all data parallel group names
    data_parallel_group_names = set()
    for param_group in param_groups:
        for param in param_group["params"]:
            if is_moe_param(param):
                data_parallel_group_names.add(param.group_name)
    data_parallel_group_names = list(data_parallel_group_names)
    group_moe = {}
    # Create the param MoE groups, leave param assign to next step
    for param_group in param_groups:
        group_moe[param_group["name"]] = {}
        for key in data_parallel_group_names:
            group_moe[param_group["name"]][key] = {}
            group_moe[param_group["name"]][key]["name"] = key
            group_moe[param_group["name"]][key]["moe"] = True
            for ori_key in param_group.keys():
                if ori_key != "name":
                    if ori_key == "params":
                        group_moe[param_group["name"]][key][ori_key] = []
                    else:
                        group_moe[param_group["name"]][key][ori_key] = param_group[ori_key]
    # Assign param
    for param_group in param_groups:
        new_params = []
        for param in param_group["params"]:
            if is_moe_param(param):
                group_moe[param_group["name"]][param.group_name]["params"].append(param)
                # param_group['params'].remove(param)
            else:
                new_params.append(param)
        param_group["params"] = new_params

    # Flatten the moe groups
    if max_group_size is not None:
        for _, v in group_moe.items():
            for _, v1 in v.items():
                cur_group = []
                all_groups = []
                size_of_cur_group = 0
                for param in v1["params"]:
                    if size_of_cur_group + param.numel() <= max_group_size:
                        cur_group.append(param)
                        size_of_cur_group += param.numel()
                    else:
                        all_groups.append(cur_group)
                        cur_group = [param]
                        size_of_cur_group = param.numel()
                if cur_group:
                    all_groups.append(cur_group)
                for group in all_groups:
                    new_dict = {}
                    for key, val in v1.items():
                        if key != "params":
                            new_dict[key] = val
                    new_dict["params"] = group
                    param_groups.append(new_dict)
    else:
        for _, v in group_moe.items():
            for _, v1 in v.items():
                param_groups.append(v1)

    return tuple(param_groups)


def create_moe_param_groups(model, weight_decay):
    parameters = {"params": list(model.parameters()), "name": "default", "weight_decay": weight_decay}

    return split_params_into_different_moe_groups_for_optimizer(parameters)
