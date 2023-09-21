from typing import Dict, Tuple

from internlm.core.context.parallel_context import global_context as gpc
from internlm.model.utils import is_gate_param, is_moe_param, is_norm_param


def split_params_into_different_groups_for_optimizer(param_groups: Tuple[Dict]) -> Tuple[Dict]:
    """Split parameters into different MoE groups for optimizer
    Compatiable with muiltiple param groups, each should have a name

    Args:
        param_groups (Tuple[Dict]): The list of parameter groups to split
        Output Example:
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

    new_groups = {}
    for pgroup in param_groups:
        new_groups[pgroup["name"]] = {}

        # create new groups for gate and norm
        for key in ["gate", "norm"]:
            new_groups[pgroup["name"]][key] = {}
            new_groups[pgroup["name"]][key]["name"] = key
            new_groups[pgroup["name"]][key][key] = True
        # create moe groups
        for key in gpc.expert_parallel_group_names:
            new_groups[pgroup["name"]][key] = {}
            new_groups[pgroup["name"]][key]["name"] = key
            new_groups[pgroup["name"]][key]["moe"] = True

        # copy attribute from origin group
        for ori_key in pgroup.keys():
            for key in new_groups[pgroup["name"]].keys():
                if ori_key != "name":
                    if ori_key == "params":
                        new_groups[pgroup["name"]][key][ori_key] = []
                    else:
                        new_groups[pgroup["name"]][key][ori_key] = pgroup[ori_key]
        # Assign param
        origin_params = []
        for param in pgroup["params"]:
            if is_moe_param(param):
                new_groups[pgroup["name"]][param.group_name]["params"].append(param)
            elif is_norm_param(param):
                new_groups[pgroup["name"]]["norm"]["params"].append(param)
            elif is_gate_param(param):
                new_groups[pgroup["name"]]["gate"]["params"].append(param)
            else:
                origin_params.append(param)

        pgroup["params"] = origin_params

    for _, v in new_groups.items():
        for _, v1 in v.items():
            param_groups.append(v1)

    return tuple(param_groups)


def create_param_groups(model, weight_decay):
    parameters = {"params": list(model.parameters()), "name": "default", "weight_decay": weight_decay}

    return split_params_into_different_groups_for_optimizer(parameters)
