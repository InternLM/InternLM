from typing import Dict, Tuple

from internlm.core.context.parallel_context import global_context as gpc


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

    def _get_group(param):
        group_keys = ["is_expert", "is_gate", "is_norm"]
        for i, key in enumerate(group_keys):
            if hasattr(param, key) and getattr(param, key):
                # experts param should return its group name
                if i == 0:
                    return param.group_name
                else:
                    return key[3:]
        # TODO: deal with fp32 group
        return None

    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    new_groups = []
    for pgroup in param_groups:
        current_groups = {}

        # create new groups for gate and norm
        for key in ["gate", "norm"]:
            current_groups[key] = {"name": key, key: True, "params": []}
        # create moe groups
        for key in gpc.expert_parallel_group_names:
            current_groups[key] = {"name": key, "moe": True, "params": []}

        # copy attribute from origin group
        for ori_key in pgroup.keys():
            if ori_key not in ("name", "params"):
                for _, group in current_groups.items():
                    group[ori_key] = pgroup[ori_key]

        # Assign param
        origin_params = []
        for param in pgroup["params"]:
            group = _get_group(param)
            if group is not None:
                current_groups[group]["params"].append(param)
            else:
                origin_params.append(param)

        pgroup["params"] = origin_params

        new_groups.append(current_groups)

    for g in new_groups:
        for _, v in g.items():
            param_groups.append(v)

    return tuple(param_groups)


def create_param_groups(model, weight_decay):
    parameters = {"params": list(model.parameters()), "name": "default", "weight_decay": weight_decay}

    return split_params_into_different_groups_for_optimizer(parameters)
