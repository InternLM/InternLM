from typing import Dict, Tuple

import torch


def split_params_into_different_groups_for_optimizer(param_groups: Tuple[Dict]) -> Tuple[Dict]:
    """Split parameters into different groups for optimizer
    Compatiable with muiltiple param groups, each should have a name

    Args:
        param_groups (Tuple[Dict]): The list of parameter groups to split
        Input Example:
        >>> (
        >>>     {'name': 'default', 'params': [tensor], 'weight_decay' :xxx},
        >>> )

    Returns:
        Tuple[Dict]: list of params groups for optimizer
        Output Example:
        >>> (
        >>>     {'name': 'default','params': [tensor],'weight_decay' :xxx},
        >>>     {'name': 'default_fp32', 'params': [tensor],'weight_decay' :xxx},
        >>>     ...,
        >>> )

    Returns:
        Tuple[Dict]: list of fp16/fp32 groups for optimizer
    """
    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    fp32_group = {"name": "fp32", "params": []}
    for pgroup in param_groups:
        # copy attribute from origin group, we assume the input param_groups only
        # have one group, so the attribute will not be copyed multiple times.
        for ori_key in pgroup.keys():
            if ori_key not in ("name", "params"):
                fp32_group[ori_key] = pgroup[ori_key]
        # Assign param
        origin_params = []
        for param in pgroup["params"]:
            if param.dtype == torch.float32:
                fp32_group["params"].append(param)
            else:
                origin_params.append(param)
        # bf16 param group, the first group in the param_groups
        pgroup["params"] = origin_params

    param_groups.append(fp32_group)

    return tuple(param_groups)


def create_param_groups(model, weight_decay):
    parameters = {"params": list(model.parameters()), "name": "default", "weight_decay": weight_decay}

    return split_params_into_different_groups_for_optimizer(parameters)
