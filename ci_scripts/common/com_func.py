#!/usr/bin/env python
# -*- encoding: utf-8 -*-


def merge_dicts(dict_a: dict, dict_b: dict):
    for key in dict_b.keys():
        if isinstance(dict_b[key], dict):
            dict_b[key] = {**dict_a[key], **dict_b[key]}
            merge_dicts(dict_a[key], dict_b[key])
    dict_c = {**dict_a, **dict_b}
    return dict_c


def format_dict_to_py_string(data: dict, indent=0, is_nested=False):
    result = ""
    for key, value in data.items():
        if isinstance(value, dict):
            result += f"{' ' * indent}{key} = dict(\n"
            result += format_dict_to_py_string(value, indent + 4, is_nested=True)
            result += f"{' ' * indent})"
        else:
            result += f"{' ' * indent}{key} = {repr(value)}"
        if is_nested:
            result += ","
        result += "\n"
    result = f"""\
{result}
"""
    return result
