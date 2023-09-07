#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context

import inspect
import random
import socket
import sys
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist

from internlm.utils.common import SingletonMeta
from internlm.utils.logger import get_logger
from internlm.utils.timeout import LLM_NCCL_TIMEOUT

from . import process_group_initializer as pgroup_initializer
from .process_group_initializer import ParallelMode
from .random import add_seed, get_seeds, set_mode

IS_TENSOR_PARALLEL = "is_tensor_parallel"

logger = get_logger(__file__)


class Config(dict):
    """This is a wrapper class for dict objects so that values of which can be
    accessed as attributes.

    Args:
        config (dict): The dict object to be wrapped.
    """

    def __init__(self, config: dict = None):  # pylint: disable=W0231
        if config is not None:
            for k, v in config.items():
                self._add_item(k, v)

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, key):
        try:
            value = super().__getitem__(key)
            return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    def _add_item(self, key, value):
        if isinstance(value, dict):
            self.__setattr__(key, Config(value))
        else:
            self.__setattr__(key, value)

    def update(self, config):
        assert isinstance(config, (Config, dict)), "can only update dictionary or Config objects."
        for k, v in config.items():
            self._add_item(k, v)
        return self

    @staticmethod
    def from_file(filename: str):
        """Reads a python file and constructs a corresponding :class:`Config` object.

        Args:
            filename (str): Name of the file to construct the return object.

        Returns:
            :class:`Config`: A :class:`Config` object constructed with information in the file.

        Raises:
            AssertionError: Raises an AssertionError if the file does not exist, or the file is not .py file
        """

        # check config path
        if isinstance(filename, str):
            filepath = Path(filename).absolute()
        elif isinstance(filename, Path):
            filepath = filename.absolute()

        assert filepath.exists(), f"{filename} is not found, please check your configuration path"

        # check extension
        extension = filepath.suffix
        assert extension == ".py", "only .py files are supported"

        # import the config as module
        remove_path = False
        if filepath.parent not in sys.path:
            sys.path.insert(0, (filepath))
            remove_path = True

        module_name = filepath.stem
        source_file = SourceFileLoader(fullname=str(module_name), path=str(filepath))
        module = source_file.load_module()  # pylint: disable=W4902,E1120,W1505

        # load into config
        config = Config()

        for k, v in module.__dict__.items():
            if k.startswith("__") or inspect.ismodule(v) or inspect.isclass(v):
                continue
            else:
                config._add_item(k, v)

        # remove module
        del sys.modules[module_name]
        if remove_path:
            sys.path.pop(0)

        return config


class ParallelContext(metaclass=SingletonMeta):
    """This class provides interface functions for users to get the parallel context,
    such as the global rank, the local rank, the world size, etc. of each device.

    """

    def __init__(self):
        # distributed settings
        self._global_ranks = dict()
        self._local_ranks = dict()
        self._world_sizes = dict()
        self._groups = dict()
        self._cpu_groups = dict()
        self._ranks_in_group = dict()

        # load config from file
        self._config = None

        # default parallel args, will be overwritten during process group intialization
        self.world_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.zero1_parallel_size = -1
        self.nettest_parallel_size = 1
        self.num_processes_on_current_node = -1
        self.virtual_pipeline_parallel_size = None
        self.virtual_pipeline_parallel_rank = None

    @property
    def config(self):
        return self._config

    def load_config(self, config: Union[dict, str]):
        """Loads the configuration from either a dict or a file.

        Args:
            config (dict or str): Either a dict containing the configuration information or the filename
                of a file containing the configuration information.

        Raises:
            TypeError: Raises a TypeError if `config` is neither a dict nor a str.
        """
        if isinstance(config, str):
            self._config = Config.from_file(config)
        elif isinstance(config, dict):
            self._config = Config(config)
        else:
            raise TypeError("Invalid type for config, only dictionary or string is supported")

    def detect_num_processes_on_current_node(self):
        hostname = socket.gethostname()
        hostname_list = [None for _ in range(self.get_world_size(ParallelMode.GLOBAL))]
        dist.all_gather_object(hostname_list, hostname, group=self.get_group(ParallelMode.GLOBAL))
        counter = Counter(hostname_list)
        self.num_processes_on_current_node = counter[hostname]

    @staticmethod
    def _check_parallel_mode(parallel_mode: ParallelMode):
        assert isinstance(
            parallel_mode, ParallelMode
        ), f"expected the argument parallel_mode to be of enum ParallelMode, but got {type(parallel_mode)}"

    def get_global_rank(self):
        """Returns the global rank of the current device.

        Returns:
            int: The global rank of the current device
        """
        return self._global_ranks[ParallelMode.GLOBAL]

    def get_local_rank(self, parallel_mode: ParallelMode):
        """Returns the local rank of the current device.

        Args:
            parallel_mode: The parallel mode for the rank.

        Returns:
            int: The local rank of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._local_ranks.get(parallel_mode, 0)

    def get_next_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the next device.

        Args:
            parallel_mode: The parallel mode for the rank.

        Returns:
            int: The global rank of the next device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def get_prev_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the previous device.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The global rank of the previous device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank - 1) % world_size]

    def is_using_dp(self):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.DATA and its world_size is greater than 1.
        """
        return self.is_initialized(ParallelMode.DATA) and self.get_world_size(ParallelMode.DATA) > 1

    def is_using_tp(self):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.TENSOR and its world_size is greater than 1.
        """
        return self.is_initialized(ParallelMode.TENSOR) and self.get_world_size(ParallelMode.TENSOR) > 1

    def is_using_pp(self):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.PIPELINE and its world_size is greater than 1.
        """
        return self.is_initialized(ParallelMode.PIPELINE) and self.get_world_size(ParallelMode.PIPELINE) > 1

    def is_using_sequence(self):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.SEQUENCE and its world_size is greater than 1.
        """
        return False
        # return gpc.is_initialized(ParallelMode.SEQUENCE) and gpc.get_world_size(ParallelMode.SEQUENCE) > 1

    def is_first_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the first one
        among its group for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = 0
        if self.is_initialized(parallel_mode):
            rank = self.get_local_rank(parallel_mode)
        return rank == 0

    def is_rank_for_log(self):
        """Returns a boolean value indicating whether the current device should print log."""
        is_log_rank = (
            self.is_first_rank(ParallelMode.DATA)
            and self.is_first_rank(ParallelMode.TENSOR)
            and self.is_last_rank(ParallelMode.PIPELINE)
        )
        return is_log_rank

    def is_last_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the last one
        among its group for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = 0
        world_size = 1
        if self.is_initialized(parallel_mode):
            rank = self.get_local_rank(parallel_mode)
            world_size = self.get_world_size(parallel_mode)
        return rank == world_size - 1

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self.virtual_pipeline_parallel_size is not None and self.virtual_pipeline_parallel_rank != 0:
                return False
        return self.is_first_rank(ParallelMode.PIPELINE)

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if (
                self.virtual_pipeline_parallel_size is not None
                and self.virtual_pipeline_parallel_rank != self.virtual_pipeline_parallel_size - 1
            ):
                return False
        return self.is_last_rank(ParallelMode.PIPELINE)

    def get_world_size(self, parallel_mode: ParallelMode):
        """Returns the world size for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The world size for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._world_sizes.get(parallel_mode, 1)

    def get_group(self, parallel_mode: ParallelMode):
        """Returns the group of the current device for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            torch.distributed.ProcessGroup: The group of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._groups[parallel_mode]

    def get_ranks_in_group(self, parallel_mode: ParallelMode):
        """Returns the rank of the current device for `parallel_mode` in the group.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The rank of the current device for `parallel_mode` in the group.
        """
        self._check_parallel_mode(parallel_mode)
        return self._ranks_in_group[parallel_mode]

    def get_cpu_group(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        return self._cpu_groups[parallel_mode]

    def init_global_dist(self, rank: int, world_size: int, backend: str, host: str, port: int, use_cpu: bool = False):
        """Initializes the global distributed environment

        Args:
           rank (int): rank for the default process group.
           world_size (int): world size of the default process group.
           backend (str): backend for ``torch.distributed``
           host (str): the master address for distributed training.
           port (str): the master port for distributed training.
           use_cpu (bool): whether to set up cpu process group.
        """
        # initialize the default process group
        init_method = f"tcp://[{host}]:{port}"
        dist.init_process_group(
            rank=rank,
            world_size=world_size,
            backend=backend,
            init_method=init_method,
            timeout=LLM_NCCL_TIMEOUT,
        )

        # None will give the default global process group for pytorch dist operations
        ranks = list(range(world_size))
        if use_cpu:
            cpu_group = (
                dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                if dist.get_backend() != "gloo"
                else None
            )
        else:
            cpu_group = None
        self._register_dist(rank, world_size, dist.GroupMember.WORLD, cpu_group, ranks, ParallelMode.GLOBAL)
        self._global_ranks[ParallelMode.GLOBAL] = rank

    def _register_dist(self, local_rank, world_size, process_group, cpu_group, ranks_in_group, mode):
        self._check_parallel_mode(mode)
        self._local_ranks[mode] = local_rank
        self._world_sizes[mode] = world_size
        self._groups[mode] = process_group
        self._cpu_groups[mode] = cpu_group
        self._ranks_in_group[mode] = ranks_in_group

    def check_sanity(self):
        """Checks sanity of the parallel context.

        Raises:
            AssertionError: Raises an AssertionError if the world size does not equal to the product
                of data parallel size, pipeline parallel size and tensor parallel size.
        """
        dps = self.data_parallel_size
        pps = self.pipeline_parallel_size
        tps = self.tensor_parallel_size
        ws = self.world_size
        assert ws == dps * pps * tps, (
            f"Expected the world size {ws} to be equal to data"
            f" parallel size ({dps}) * pipeline parallel size "
            f"({pps}) * tensor parallel size ({tps})"
        )
        assert self.zero1_parallel_size > 0
        assert self.data_parallel_size % self.zero1_parallel_size == 0

    def _set_parallel_size_from_config(self, config: dict, key: str, attr_name: str):
        if key in config:
            ele = config[key]
            if isinstance(ele, int):
                setattr(self, attr_name, ele)
            elif isinstance(ele, dict):
                setattr(self, attr_name, ele["size"])
            else:
                raise NotImplementedError(
                    f'{"Parallel configuration does not support this kind of argument, please use int or dict"}'
                )

    def init_parallel_groups(self):
        """Initializes the parallel groups."""

        # get rank and world size
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)
        self.world_size = world_size

        # set parallel size as attributes for global context
        parallel_config = self.config.get("parallel", None)
        if parallel_config is not None:
            self._set_parallel_size_from_config(parallel_config, "pipeline", "pipeline_parallel_size")
            self._set_parallel_size_from_config(parallel_config, "tensor", "tensor_parallel_size")
            self._set_parallel_size_from_config(parallel_config, "zero1", "zero1_parallel_size")

        # the user should not set the data parallel size manually
        # instead, it should be calculated based on other parallel config
        self.data_parallel_size = self.world_size // (self.pipeline_parallel_size * self.tensor_parallel_size)

        # the recommended nettest_parallel_size is 32 GPUs
        self.nettest_parallel_size = 32

        if self.zero1_parallel_size <= 0:
            self.zero1_parallel_size = self.data_parallel_size

        self.check_sanity()

        initializer_args = [
            rank,
            world_size,
            self.data_parallel_size,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.zero1_parallel_size,
            self.nettest_parallel_size,
        ]

        # run initialization of different process groups
        initializers = []
        initializers.append(pgroup_initializer.Initializer_Data(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Model(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Tensor(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Zero1(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Nettest(*initializer_args))
        if self.pipeline_parallel_size > 1:
            initializers.append(pgroup_initializer.Initializer_Pipeline(*initializer_args))
        for initializer in initializers:
            parallel_setting = initializer.init_dist_group()
            if isinstance(parallel_setting, list):
                for args in parallel_setting:
                    self._register_dist(*args)
            else:
                self._register_dist(*parallel_setting)

    def is_initialized(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether `parallel_mode` is initialized
        in the current system.
        """
        return parallel_mode in self._groups

    def destroy(self):
        """Destroys the current distributed parallel environment."""
        for mode, group in self._groups.items():
            if mode is not ParallelMode.GLOBAL:
                dist.destroy_process_group(group)
        # destroy global process group
        dist.destroy_process_group()
        self._groups.clear()

    def set_device(self, device_ordinal: int = None):
        """Sets distributed processes to be bound to devices.

        Args:
           device_ordinal (int, optional): the device id to be bound to
        """
        global_rank = self.get_global_rank()
        if device_ordinal is None:
            devices_per_node = torch.cuda.device_count()
            device_ordinal = global_rank % devices_per_node

        torch.cuda.set_device(device_ordinal)
        logger.info(f"process rank {global_rank} is bound to host:{socket.gethostname()} device: {device_ordinal}")

    def set_seed(self, seed: int, dpseed_with_tpoffset: bool = False):
        """Sets seeds for all random libraries.

        Args:
            seed (int): seed for random states
        """
        pipeline_offset = self._local_ranks.get(ParallelMode.PIPELINE, 0)
        global_rank = self.get_global_rank()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        assert torch.cuda.is_available()

        # data parallel seed are kept the same in the same pipeline stage
        dp_seed = seed
        if dpseed_with_tpoffset:
            dp_seed = seed + pipeline_offset * 1024
        add_seed(ParallelMode.DATA, dp_seed)
        add_seed(ParallelMode.DUMMY, dp_seed)

        # model parallel seeds are different across ranks
        if self.is_initialized(ParallelMode.TENSOR):
            tp_rank = self.get_local_rank(ParallelMode.TENSOR)
            tp_seed = seed + tp_rank + pipeline_offset * 1024
            add_seed(ParallelMode.TENSOR, tp_seed)

        # we do not set the random state mode to ParallelMode.DATA until model is built (instead, we use a dummy mode
        # during model construction), this is because the random state will be different in different tensor parallel
        # device of the same data parallel group. The underlying reason is that the device of tp_rank = 0 will perform
        # additional random operations during the RowParallelLinear module building process.
        set_mode(ParallelMode.DUMMY)

        seeds = get_seeds()
        seed_str = ", ".join([f"{k}: {v}" for k, v in seeds.items()])
        logger.info(
            f"initialized seed on rank {global_rank}, "
            f"numpy: {seed}, python random: {seed}, {seed_str},"
            f"the default parallel seed is {ParallelMode.DATA}."
        )

    def set_virtual_pipeline_parallel_size(self, size):
        self.virtual_pipeline_parallel_size = size

    def set_virtual_pipeline_parallel_rank(self, rank):
        self.virtual_pipeline_parallel_rank = rank


global_context = ParallelContext()
