#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import json
import os

from ci_scripts.common import com_func
from internlm.core.context import Config


def generate_new_config(config_py_file, test_config_json, case_name):
    # generate path of the new config py
    config_path = os.path.split(config_py_file)
    new_config_py_file = os.path.join(config_path[0], case_name + ".py")

    # merge dict
    origin_config = Config.from_file(config_py_file)
    with open(test_config_json) as f:
        test_config = json.load(f)
    if test_config:
        if case_name not in test_config.keys():
            raise KeyError(f"the {case_name} doesn't exist.Please check {test_config} again!")
    new_config = com_func.merge_dicts(origin_config, test_config[case_name])
    print(f"new config is:\n{new_config}")

    # write new config to py file
    file_content = com_func.format_dict_to_py_string(new_config)
    with open(new_config_py_file, "w") as f:
        f.write(file_content)
    print(f"The new test train config file is {new_config_py_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin_config",
        type=str,
        default="./ci_scripts/train/ci_7B_sft.py",
        help="path to the origin train config file",
    )
    parser.add_argument(
        "--test_config",
        type=str,
        default="./ci_scripts/train/test_config.json",
        help="path to the test train config file",
    )
    parser.add_argument("--case_name", type=str, help="name of the case which will be runned ")
    args = parser.parse_args()
    generate_new_config(args.origin_config, args.test_config, args.case_name)
