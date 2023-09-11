#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Dict, Union

import torch

from internlm.core.context import Config
from internlm.core.context import global_context as gpc
from internlm.monitor import initialize_light_monitor
from internlm.utils.common import get_master_node
from internlm.utils.logger import get_logger
from internlm.utils.timeout import llm_timeout

logger = get_logger(__file__)


def get_default_parser():
    """Reads user command line and uses an argument parser to parse the input arguments.
    Input arguments include configuration, host, port, world size, local rank, backend for torch.distributed.

    Returns:
       Parser: Returns the parser with the default arguments, the user may add customized arguments into this parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the config file")
    parser.add_argument(
        "--launcher",
        type=str,
        default="slurm",
        choices=["slurm", "torch"],
        help="launcher for launching distributed environment",
    )
    parser.add_argument("--host", type=str, help="the master address for distributed training")
    parser.add_argument("--port", type=int, default=8888, help="the master port for distributed training")
    parser.add_argument("--world_size", type=int, help="world size for distributed training")
    parser.add_argument("--rank", type=int, help="rank for the default process group")
    parser.add_argument("--local_rank", type=int, help="local rank on the node")
    parser.add_argument("--backend", type=str, default="nccl", help="backend for distributed communication")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--profiling", default=False, action="store_true", help="enable/disable profiling.")
    return parser


def args_sanity_check():
    assert gpc.config is not None, "config is not load!"

    # the default model type is INTERNLM
    if "model_type" not in gpc.config:
        gpc.config._add_item("model_type", "INTERNLM")

    # procssing the parallel config in gpc
    if "zero1" not in gpc.config.parallel:
        gpc.config.parallel._add_item("zero1", -1)

    if "pipeline" not in gpc.config.parallel:
        gpc.config.parallel._add_item("pipeline", 1)

    if "tensor" not in gpc.config.parallel:
        gpc.config.parallel._add_item("tensor", 1)

    # processing the data config in gpc
    data = gpc.config.data

    assert data.seq_len is not None, "'seq_len' must be given a value"
    assert data.micro_bsz is not None, "'micro_bsz' must be given a value"

    if "packed_length" in data and gpc.is_rank_for_log():
        logger.warning("packed_length would be ignored and will be setted as seq_len * micro_bsz.")

    data._add_item("packed_length", data.seq_len * data.micro_bsz)

    if "micro_num" not in data:
        data._add_item("micro_num", 1)

    data._add_item("gradient_accumulation", data.micro_num)
    if gpc.is_rank_for_log():
        logger.info(f"gradient_accumulation size will be setted to {data.micro_num}.")

    # batch_size should be equal with micro_num, should not use it directly
    data._add_item("batch_size", data.micro_num)

    if "min_length" not in data:
        data._add_item("min_length", 0)

    if "train_folder" not in data:
        data._add_item("train_folder", None)

    if "valid_folder" not in data:
        data._add_item("valid_folder", None)

    if "valid_micro_num" not in data:
        data._add_item("valid_micro_num", data.micro_num)

    if "valid_every" not in data:
        data._add_item("valid_every", 0)

    if "empty_cache_and_diag_interval" not in data:
        data._add_item("empty_cache_and_diag_interval", 50)

    if "diag_outlier_ratio" not in data:
        data._add_item("diag_outlier_ratio", 1.1)
    data.diag_outlier_ratio = max(1, data.diag_outlier_ratio)

    if gpc.is_rank_for_log():
        logger.info("+" * 15 + " Data Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"seq_len: {data.seq_len}")
        logger.info(f"micro_num: {data.micro_num}")
        logger.info(f"micro_bsz: {data.micro_bsz}")
        logger.info(f"packed_length: {data.packed_length}")
        logger.info(f"pack_sample_into_one: {data.pack_sample_into_one}")
        logger.info(f"min_length: {data.min_length}")
        logger.info(f"valid_micro_num: {data.valid_micro_num}")
        logger.info(f"valid_every: {data.valid_every}")

    # processing the checkpoint config
    ckpt = gpc.config.ckpt
    if "enable_save_ckpt" not in ckpt:
        ckpt._add_item("enable_save_ckpt", True)

    # Saving checkpoint args.
    if ckpt.enable_save_ckpt:
        assert "checkpoint_every" in ckpt, "If enable save checkpoint, must give checkpoint_every in config.data!"
        assert ckpt.checkpoint_every > 0
        assert "save_ckpt_folder" in ckpt, "If enable save checkpoint, must give save_ckpt_folder in config.data!"

        if "async_upload" not in ckpt:
            ckpt._add_item("async_upload", False)  # async defalut is False.
        else:
            if ckpt.async_upload:
                assert "save_ckpt_folder" in ckpt
                if "boto3:" not in ckpt.save_ckpt_folder:
                    if gpc.is_rank_for_log():
                        logger.warning(
                            "Storing ckpt on file system does not support asynchronous storage, will use sync save!"
                        )
                    ckpt.async_upload = False
                else:
                    if "async_upload_tmp_folder" not in ckpt:
                        ckpt._add_item("async_upload_tmp_folder", "/dev/shm/internlm_tmp_ckpt/")

        if not ckpt.async_upload:
            ckpt._add_item("async_upload_tmp_folder", None)

        if "oss_snapshot_freq" not in ckpt:
            ckpt._add_item("oss_snapshot_freq", float("inf"))  # if oss_snapshot_freq not given, we disable.
    else:
        ckpt._add_item("checkpoint_every", float("inf"))
        ckpt._add_item("oss_snapshot_freq", float("inf"))
        ckpt._add_item("save_ckpt_folder", None)
        ckpt._add_item("async_upload", False)
        ckpt._add_item("async_upload_tmp_folder", None)
        ckpt._add_item("snapshot_ckpt_folder", None)

    if "load_ckpt_folder" not in ckpt:
        ckpt._add_item("load_ckpt_folder", None)

    if "stop_file_path" not in ckpt:
        ckpt._add_item("stop_file_path", None)

    if "auto_resume" not in ckpt:
        # If 'auto_resume' is not given, we set it to True, so internlm can have opportunity
        # to auto-load latest checkpoint.
        ckpt._add_item("auto_resume", True)

    if gpc.is_rank_for_log():
        logger.info("+" * 15 + " Ckpt Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"is enable save ckpt: {ckpt.enable_save_ckpt}")
        logger.info(f"save_ckpt_folder: {ckpt.save_ckpt_folder}")
        logger.info(f"checkpoint_every: {ckpt.checkpoint_every}")

    # tensorboard writer config
    if "enable_tb" not in gpc.config:
        gpc.config._add_item("enable_tb", True)
    if "tensorboard_folder" not in gpc.config:
        gpc.config._add_item(
            "tensorboard_folder", os.environ["tensorboard_folder"] if "tensorboard_folder" in os.environ else None
        )
    if "resume_tb_folder" not in gpc.config:
        gpc.config._add_item(
            "resume_tb_folder", os.environ["resume_tb_folder"] if "resume_tb_folder" in os.environ else None
        )

    if gpc.is_rank_for_log():
        logger.info(f"tensorboard_folder: {gpc.config.tensorboard_folder}")
        logger.info(f"resume_tb_folder: {gpc.config.resume_tb_folder}")

    # cudnn
    torch.backends.cudnn.benchmark = gpc.config.get("cudnn_benchmark", False)
    torch.backends.cudnn.deterministic = gpc.config.get("cudnn_deterministic", False)
    clip_grad_norm = gpc.config.hybrid_zero_optimizer.get("clip_grad_norm", 0.0)

    if gpc.is_rank_for_log():
        logger.info("+" * 15 + " Other Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"cudnn.benchmark: {torch.backends.cudnn.benchmark }")
        logger.info(f"cudnn.deterministic: {torch.backends.cudnn.deterministic }")
        logger.info(f"clip_grad_norm: {clip_grad_norm}")

    model = gpc.config.model
    if "dtype" not in model:
        logger.warning("dtype is not set, use torch.float16 by defalut!")
        model._add_item("dtype", torch.float16)
    else:
        if gpc.config.model.dtype == "torch.bfloat16":
            gpc.config.model.dtype = torch.bfloat16
        elif gpc.config.model.dtype in ("torch.float16", "torch.half"):
            gpc.config.model.dtype = torch.float16
        elif gpc.config.model.dtype == "torch.float32":
            gpc.config.model.dtype = torch.float32
        elif gpc.config.model.dtype == "torch.tf32":
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            gpc.config.model.dtype = torch.float32
        else:
            assert gpc.config.model.dtype in [
                "torch.float16",
                "torch.half",
                "torch.bfloat16",
                "torch.float32",
                "torch.tf32",
            ]

    if "checkpoint" in model:
        if model.checkpoint is True:
            model.checkpoint = 1
        elif model.checkpoint is False:
            model.checkpoint = 0
        else:
            assert (
                model.checkpoint >= 0 and model.checkpoint <= 1
            ), f'model.checkpoint: "{model.checkpoint}" should >=0 and <=1'

    if gpc.is_rank_for_log():
        logger.info("+" * 15 + " Model Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"Model: {gpc.config.model}")

        logger.info("+" * 15 + " grad_scaler Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"grad_scaler: {gpc.config.grad_scaler}")

        logger.info("+" * 15 + " hybrid_zero_optimizer Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"hybrid_zero_optimizer: {gpc.config.hybrid_zero_optimizer}")

        logger.info("+" * 15 + " adam Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"adam: {gpc.config.adam}")

        logger.info("+" * 15 + " beta2_scheduler Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"beta2_scheduler: {gpc.config.beta2_scheduler}")

    # process the model config
    if "use_flash_attn" not in gpc.config.model:
        gpc.config.model._add_item("use_flash_attn", True)

    # process the parallel config
    if "sequence_parallel" not in gpc.config.parallel:
        gpc.config.parallel._add_item("sequence_parallel", False)
    else:
        assert not (
            gpc.config.parallel.sequence_parallel is True and gpc.config.model.use_flash_attn is False
        ), "sequence parallel does not support use_flash_attn=False"

    # monitoring default config
    monitor_default_config = {
        "alert_address": None,  # compatible with old alert config
        "monitor": {  # new monitoring config
            "alert": {"enable_feishu_alert": False, "feishu_alert_address": None, "light_monitor_address": None}
        },
    }

    for key, value in monitor_default_config.items():
        if key not in gpc.config:
            gpc.config._add_item(key, value)

    alert = gpc.config.monitor.alert

    if alert.enable_feishu_alert and not alert.feishu_alert_address and gpc.is_rank_for_log():
        logger.warning("alert is enable but alert_address is not set")

    optim_ckpt = gpc.config.hybrid_zero_optimizer
    if "zero_overlap_communication" in optim_ckpt:
        # Compatible with the old interfaces.
        optim_ckpt._add_item("overlap_sync_grad", optim_ckpt.zero_overlap_communication)
    if "overlap_sync_grad" not in optim_ckpt:
        optim_ckpt._add_item("overlap_sync_grad", False)
    if "overlap_sync_param" not in optim_ckpt:
        optim_ckpt._add_item("overlap_sync_param", False)
    if gpc.is_rank_for_log():
        logger.info(
            f"overlap_sync_grad:{optim_ckpt.overlap_sync_grad}, overlap_sync_param:{optim_ckpt.overlap_sync_param}"
        )


def launch(
    config: Union[str, Path, Config, Dict],
    rank: int,
    world_size: int,
    host: str,
    port: int,
    backend: str = "nccl",
    local_rank: int = None,
    seed: int = 1024,
):
    """This function first parses the configuration arguments, using :func:`parse_args()` in case one of the input
    arguments are not given. Then initialize and set distributed environment by calling global_context's functions.

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        rank (int): Rank for the default process group
        world_size (int): World size of the default process group
        host (str): The master address for distributed training
        port (str): The master port for distributed training
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        local_rank (int, optional):
            Rank for the process on the node and is used to set the default CUDA device,
            defaults to None. If local_rank = None, the default device ordinal will be calculated automatically.
        seed (int, optional): Specified random seed for every process. Defaults to 1024.

    Raises:
        Exception: Raise exception when config type is wrong
    """

    # set config
    assert isinstance(
        config, (Config, str, Path, dict)
    ), f"expected argument config to be Config, str or Path, but got {type(config)}"
    if not isinstance(config, Config) and isinstance(config, dict):
        config = Config(config)
    if isinstance(config, (str, Path)):
        config = Config.from_file(config)
    gpc.load_config(config)

    # init default process group
    gpc.init_global_dist(rank, world_size, backend, host, port)

    # init process groups for different parallel modes from config
    gpc.init_parallel_groups()

    # set cuda device
    if torch.cuda.is_available():
        # if local rank is not given, calculate automatically
        gpc.set_device(local_rank)

    # set the number of processes running on the same node
    gpc.detect_num_processes_on_current_node()

    gpc.set_seed(seed)

    if gpc.is_rank_for_log():
        logger.info(
            f"Distributed environment is initialized, "
            f"data parallel size: {gpc.data_parallel_size}, pipeline parallel size: {gpc.pipeline_parallel_size}, "
            f"tensor parallel size: {gpc.tensor_parallel_size}",
        )


def launch_from_slurm(
    config: Union[str, Path, Config, Dict],
    host: str,
    port: int,
    backend: str = "nccl",
    seed: int = 1024,
):
    """A wrapper for internlm.launch for SLURM launcher by reading rank and world size from the environment variables
    set by SLURM

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        host (str): The master address for distributed training
        port (str): The master port for distributed training
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
    """
    try:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NPROCS"])
    except KeyError as e:
        raise RuntimeError(f"Could not find {e} in the SLURM environment")

    launch(
        config=config,
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        backend=backend,
        seed=seed,
    )


def launch_from_torch(
    config: Union[str, Path, Config, Dict],
    backend: str = "nccl",
    seed: int = 1024,
):
    """A wrapper for internlm.launch for torchrun or torch.distributed.launch by reading rank and world size
    from the environment variables set by PyTorch

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
    """
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])
    except KeyError as e:
        raise RuntimeError(f"Could not find {e} in the torch environment")

    launch(
        config=config,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        backend=backend,
        seed=seed,
    )


@llm_timeout(func_name="initialize_distributed_env")
def initialize_distributed_env(
    config: str,
    launcher: str = "slurm",
    master_port: int = 8888,
    seed: int = 1024,
    args_check=True,
):
    """
    Initialize distributed environment for distributed training.

    Args:
        config (str): Config file path.
        launcher (str): Launcher for launching distributed environment, can be slurm or torch. "slurm" by default.
        master_port (str): The master port for distributed training. 8888 by default.
        seed (int, optional): Specified random seed for every process. 1024 by default.
    """

    torch.cuda.empty_cache()

    if launcher == "torch":
        launch_from_torch(config=config, seed=seed)
    elif launcher == "slurm":
        launch_from_slurm(
            config=config,
            host=get_master_node(),
            port=master_port,
            seed=seed,
        )
    else:
        assert launcher in ["slurm", "torch"], "launcher only support slurm or torch"

    if args_check:
        args_sanity_check()

    # init light monitor client
    alert_config = gpc.config.monitor.alert
    if alert_config.enable_feishu_alert and gpc.is_rank_for_log():
        light_monitor_address = alert_config.light_monitor_address
        if light_monitor_address:
            initialize_light_monitor(light_monitor_address)
        else:
            logger.warning("monitor address is none, monitor could not be used!")


def get_config_value(config, key, defalut):
    try:
        value = config[key]
    except KeyError:
        value = defalut
    return value
