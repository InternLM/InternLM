from typing import Dict, Tuple

import torch


def split_params_into_different_groups_for_optimizer(param_groups: Tuple[Dict]) -> Tuple[Dict]:
    """Split parameters into different groups for optimizer
    Compatiable with muiltiple param groups, each should have a name

    Args:
        param_groups (Tuple[Dict]):
            The list of parameter groups to split

    Returns:
        Tuple[Dict]:
        list of fp16/fp32 groups for optimizer
    """
    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    # Create fp32 and moe groups and copy origin attribute
    for group_param in param_groups:
        fp32_group = {}

        # copy attribute for fp32 group
        for ori_key in group_param.keys():
            if ori_key == "name":
                fp32_group["name"] = ori_key + "_fp32"
            else:
                if ori_key == "params":
                    fp32_group[ori_key] = []
                else:
                    fp32_group[ori_key] = group_param[ori_key]

        # Assign param
        new_params = []
        for param in group_param["params"]:
            if param.dtype == torch.float32:
                fp32_group["params"].append(param)
            else:
                new_params.append(param)

        # origin group without fp32
        group_param["params"] = new_params
        # append to origin group
        param_groups.append(fp32_group)

    return tuple(param_groups)


def create_param_groups(model, weight_decay):
    parameters = {"params": list(model.parameters()), "name": "default", "weight_decay": weight_decay}

    return split_params_into_different_groups_for_optimizer(parameters)
