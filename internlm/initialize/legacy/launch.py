#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from internlm.initialize.launch import get_config_value
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


def auto_resume_sanity_check(ckpt_config):
    load_given_ckpt = get_config_value(ckpt_config, "load_given_ckpt", None)
    if load_given_ckpt is None:
        return True  # default value is True
    else:
        return not load_given_ckpt


def ckpt_info_sanity_check(ckpt_config):
    load_ckpt_folder = get_config_value(ckpt_config, "load_ckpt_folder", None)

    load_model_only_folder = get_config_value(ckpt_config, "load_model_only_folder", None)

    if load_model_only_folder is not None:
        assert (
            load_ckpt_folder is None
        ), "Detect 'load_ckpt_folder' and 'load_model_only_folder' set at the same time, \
# and 'load_given_ckpt' is True, so internlm will load from 'load_ckpt_folder'"
        return dict(path=load_model_only_folder, content=("model",), ckpt_type="internlm")
    else:
        load_optimizer = get_config_value(ckpt_config, "load_optimizer", True)

        if isinstance(load_ckpt_folder, str):
            if load_optimizer:
                return dict(path=load_ckpt_folder, content=("model", "sampler", "optimizer"), ckpt_type="internlm")
            else:
                return dict(path=load_ckpt_folder, content=("model", "sampler"), ckpt_type="internlm")
        elif load_ckpt_folder is None:
            return None
        else:
            assert f"Unsupport data type:'{type(load_ckpt_folder)}' for config.ckpt arg: 'load_ckpt_folder'"
