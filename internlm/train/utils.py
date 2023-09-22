from typing import Dict, Tuple

import torch

from internlm.core.context.parallel_context import global_context as gpc
from internlm.model.utils import is_gate_param, is_moe_param, is_norm_param


def split_params_into_different_groups_for_optimizer(param_groups: Tuple[Dict]) -> Tuple[Dict]:
    """Split parameters into different MoE groups for optimizer
    Compatiable with muiltiple param groups, each should have a name

    Args:
        param_groups (Tuple[Dict]): The list of parameter groups to split
        Input Example:
        >>> (
        >>>     {'name': 'default', 'params': [tensor], 'weight_decay' :xxx},
        >>>     ...,
        >>> )

    Returns:
        Tuple[Dict]: list of params groups for optimizer
        Output Example:
        >>> (
        >>>     {'name': 'default','params': [tensor],'weight_decay' :xxx},
        >>>     {'name': 'norm', 'norm': True, 'params': [tensor],'weight_decay' :xxx},
        >>>     {'name': 'gate', 'gate': True, 'params': [tensor],'weight_decay' :xxx},
        >>>     {'name': 'moe_ep_size_4', 'moe': True, 'params':  [tensor],'weight_decay' :xxx},
        >>>     ...,
        >>> )
    """

    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    # create new groups for fp32, norm, moe gate and moe expert
    new_groups = {}
    new_groups["fp32"] = {"name": "fp32", "params": []}
    for key in ["gate", "norm"]:
        new_groups[key] = {"name": key, "sync_tp": True, "params": []}
    for key in gpc.expert_parallel_group_names:
        new_groups[key] = {"name": key, "moe": True, "params": []}

    for pgroup in param_groups:
        # copy attribute from origin group
        for ori_key in pgroup.keys():
            if ori_key not in ("name", "params"):
                for _, group in new_groups.items():
                    group[ori_key] = pgroup[ori_key]
        # assign param
        origin_params = []
        # first split the norm and gate groups, then the fp32 group, finally moe group
        for param in pgroup["params"]:
            if is_norm_param(param):
                new_groups["norm"]["params"].append(param)
            elif is_gate_param(param):
                new_groups["gate"]["params"].append(param)
            elif param.dtype == torch.float32:
                new_groups["fp32"]["params"].append(param)
            elif is_moe_param(param):
                new_groups[param.group_name]["params"].append(param)
            else:
                origin_params.append(param)
        # bf16 param group, which is the first group in the param groups
        pgroup["params"] = origin_params

    param_groups.extend(new_groups.values())

    return tuple(param_groups)


def create_param_groups(model, weight_decay):
    parameters = {"params": list(model.parameters()), "name": "default", "weight_decay": weight_decay}
    return split_params_into_different_groups_for_optimizer(parameters)
