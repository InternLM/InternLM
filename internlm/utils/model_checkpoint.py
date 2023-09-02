#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import fcntl
import os
import socket
import time
from enum import Enum
from typing import Dict

import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.trainer import TrainState
from internlm.monitor import send_alert_message
from internlm.solver.optimizer import HybridZeroOptimizer
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.storage_manager import (
    get_fns,
    get_storage_manager,
    init_storage_manager,
    llm_load,
    llm_save,
    try_get_storage_backend,
)

logger = get_logger(__file__)


class CheckpointSaveType(Enum):
    NORMAL_CHECKPOINT = 1
    SNAPSHOT_CHECKPOINT = 2


class CheckpointLoadType(Enum):
    INTERNLM = 1


class CheckpointLoadMask:
    LOAD_MODEL = 0b0001
    LOAD_OPTIMIZER = 0b0010
    LOAD_SCHEDULER = 0b0100
    LOAD_SAMPLER = 0b1000
    LOAD_ALL = 0b1111


def get_config_value(config, key, defalut):
    try:
        value = config[key]
    except KeyError:
        value = defalut
    return value


def get_model_topology(model):
    """
    Returns:
        {
            '{name}': {'dim': int}
        }
        where name is the name of the module, and all parameters under this module are
        concatenated along the dimension 'dim'.
    """

    from flash_attn.modules.embedding import VocabParallelEmbedding

    topos = {}
    for name, module in model.named_modules():
        # If it does not meet these conditions, it is shared between various tp/dp, and it is necessary to assert
        # that they are consistent.
        if isinstance(module, VocabParallelEmbedding):
            topos[name] = {"dim": 0}
    return topos


def save_model_checkpoint(folder, model):
    """
    Save the model according to the relationship between tp and dp. The principle is that the data of each tp
    will not be gathered and saved separately, which is equivalent to actual sharding. The saved weight is named
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.

    Args:
        folder: The folder to save the model
        model: The model to be saved
    """

    states = model.state_dict()
    topo = get_model_topology(model)

    if folder is not None:
        dp_size = gpc.get_world_size(ParallelMode.DATA)
        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        dp_rank = gpc.get_local_rank(ParallelMode.DATA)
        tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

        # TODO In theory, we should also consider pp level, but since pp is generally a state across machines,
        # even if pp is not considered, it will definitely not be written on the same machine.
        should_save_rank_pair = set()  # (tp_rank, dp_rank)
        for i in range(tp_size):
            should_save_rank_pair.add((i, i % dp_size))

        if (tp_rank, dp_rank) in should_save_rank_pair:
            fn = f"model_tp{tp_rank}_pp{pp_rank}.pt"
            fp = os.path.join(folder, fn)
            llm_save(fp, saved_obj=states)
            topo_fn = f"topo_tp{tp_rank}_pp{pp_rank}.json"
            topo_fp = os.path.join(folder, topo_fn)
            llm_save(topo_fp, saved_obj=topo)

    torch.distributed.barrier()


def load_model_checkpoint(folder, model):
    """
    There should be weights with names similar to the following under the folder.
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.
    """

    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    fns = get_fns(folder)
    max_pp, max_tp = 0, 0
    for fn in fns:
        if fn.startswith("model_t") and not fn.endswith(".md5"):
            segements = os.path.splitext(fn)[0].split("_")
            max_pp = max(max_pp, int(segements[-1][2:]))
            max_tp = max(max_tp, int(segements[-2][2:]))

    assert (
        pp_size == max_pp + 1
    ), f"The weights are save for {max_pp+1} pipelines, while current has {pp_size} pipelines"
    assert (
        tp_size == max_tp + 1
    ), f"The weights are save for {max_tp+1} parallelism, while current has {tp_size} tensor parallelism"

    should_load_name = f"model_tp{tp_rank}_pp{pp_rank}.pt"
    fp = os.path.join(folder, should_load_name)
    states = llm_load(fp, map_location=get_current_device())

    missing_k, unexpected_keys = model.load_state_dict(states, strict=False)
    if len(missing_k) != 0:
        logger.warning(f"Warning: missing keys {missing_k}")
    if len(unexpected_keys) != 0:
        logger.warning(f"Warning: unexpected keys {unexpected_keys}")

    # avoid to cuda oom, Ref: https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/11
    del states
    torch.cuda.empty_cache()


def save_optimizer_checkpoint(optim, state_path):
    """Store the state of the optimizer to the local file system or remote OSS.

    Args:
        optim (Optimizer)
        state_path (str): The state loading path of optimizer.
    """

    # TODO sanity check for optimizer type
    zero_rank = gpc.get_local_rank(ParallelMode.ZERO1)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
    fp = f"optimizer_tp{tp_rank}_pp{pp_rank}_zo{zero_rank}.pt"

    states = optim.state_dict()
    if isinstance(optim, HybridZeroOptimizer):
        if gpc.get_global_rank() < optim.zero_world_size * tp_size * pp_size:
            llm_save(os.path.join(state_path, fp), states)
            if "zero_devide_optim_plan" in states:
                params_per_rank_id_dict = states.pop("zero_devide_optim_plan")
                fp_meta = os.path.join(state_path, optim.rank_unique_id)
                llm_save(fp_meta, params_per_rank_id_dict)
    else:
        llm_save(os.path.join(state_path, fp), states)


def load_optimizer_checkpoint(folder, optim):
    """Load the optimizer state from the local file system or remote
    object storage Service (OSS).

    Args:
        optim (Optimizer): optimizer
        folder (str): The FS/OSS path where the optimizer will be stored.
    """

    fns = get_fns(folder)
    max_tp, max_pp, max_zero = 0, 0, 0
    for fn in fns:
        if fn.startswith("optimizer_") and not fn.endswith(".md5"):
            _, tp, pp, zero = os.path.splitext(fn)[0].split("_")
            max_zero = max(max_zero, int(zero[2:]))
            max_tp = max(max_tp, int(tp[2:]))
            max_pp = max(max_pp, int(pp[2:]))

    zero_size = gpc.get_world_size(ParallelMode.ZERO1)
    zero_rank = gpc.get_local_rank(ParallelMode.ZERO1)
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)

    assert (
        zero_size == max_zero + 1
    ), f"The weights are save for {max_zero+1} data parallel, while current has {zero_size} zero broadcast range."
    assert (
        pp_size == max_pp + 1
    ), f"The weights are save for {max_pp+1} pipelines, while current has {pp_size} pipelines"
    assert (
        tp_size == max_tp + 1
    ), f"The weights are save for {max_tp+1} parallelism, while current has {tp_size} tensor parallelism"

    fp = f"optimizer_tp{gpc.get_local_rank(ParallelMode.TENSOR)}_"
    fp += f"pp{gpc.get_local_rank(ParallelMode.PIPELINE)}_"
    fp += f"zo{zero_rank}.pt"
    states = llm_load(os.path.join(folder, fp), map_location=get_current_device())

    if isinstance(optim, HybridZeroOptimizer):
        fp_meta = os.path.join(folder, optim.rank_unique_id)
        try:
            zero_devide_optim_plan = llm_load(fp_meta)
            states.update({"zero_devide_optim_plan": zero_devide_optim_plan})
        except Exception as e:
            logger.warning(
                f"Read zero optimzer split file '{fp_meta}', for '{e}'"
                f"Please check whether loading ckpts are saved with the HybridZeroOptimizer."
            )

    optim.load_state_dict(states)
    del states
    torch.cuda.empty_cache()


def load_sampler(ckpt_path: str, sampler):
    sampler_states = llm_load(os.path.join(ckpt_path, "sampler.pt"))
    sampler.load_state_dict(sampler_states)
    if gpc.is_rank_for_log():
        pstate = copy.deepcopy(sampler_states)
        pstate.pop("indices")
        pstate.pop("rng_state")
        logger.info(f"reload sampler_states:{pstate}")
    torch.cuda.empty_cache()


def load_context(ckpt_path: str, train_state: TrainState):
    context_stuffs = llm_load(os.path.join(ckpt_path, "context.pt"))
    train_state.load_state_dict(context_stuffs)
    if gpc.is_rank_for_log():
        logger.info(f"reload train_state:{train_state}")
    torch.cuda.empty_cache()


def load_scheduler(ckpt_path: str, lr_scheduler, optimizer, train_state: TrainState):
    learning_rate = train_state.lr
    scheduler_states = llm_load(os.path.join(ckpt_path, "schedulder.pt"))
    if learning_rate != scheduler_states["base_lrs"][0] and gpc.is_rank_for_log():
        logger.warning(
            f"Using new learning rate {learning_rate} to replace old learn rate {scheduler_states['base_lrs'][0]}."
        )

    base_lrs = copy.deepcopy(scheduler_states["base_lrs"])
    scheduler_states["base_lrs"] = [learning_rate] * len(scheduler_states["base_lrs"])
    if "after_scheduler_dict" in scheduler_states:
        scheduler_states["after_scheduler_dict"]["base_lrs"] = [learning_rate] * len(
            scheduler_states["after_scheduler_dict"]["base_lrs"]
        )

    lr_scheduler.load_state_dict(scheduler_states)
    lr_scheduler.last_epoch = train_state.step_count + 1

    ratios = [learning_rate / lr for lr in base_lrs]
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = param_group["lr"] * ratios[idx]
    torch.cuda.empty_cache()

    if gpc.is_rank_for_log():
        logger.info(f"reload load_scheduler:{lr_scheduler}")


class CheckpointManager:
    """StorageManagerContext"""

    def __init__(
        self,
        ckpt_config,
        model,
        train_dl=None,
        optimizer=None,
        lr_scheduler=None,
        model_config=None,
        model_config_file=None,
        feishu_address=None,
    ) -> None:
        """
        CheckpointManager is used to decide when to store ckpt. If it is an asynchronous
        upload mode, you must call wait_async_upload_finish at the end of the program to wait
        for the asynchronous ckpt upload to complete.

        Args:
            ckpt_config (dict): model checkpoint config.
            model (nn.module): model obj
            optimizer (object): optimzier obj.
            lr_scheduler (object): lr_scheduler obj.
            model_config (dict): model config.
        """
        self.enable_save_ckpt = get_config_value(ckpt_config, "enable_save_ckpt", False)
        self.checkpoint_every = get_config_value(ckpt_config, "checkpoint_every", 1)
        self.save_ckpt_folder = get_config_value(ckpt_config, "save_ckpt_folder", None)
        self.oss_snapshot_freq: int = get_config_value(ckpt_config, "oss_snapshot_freq", 1)
        self.stop_file_path = get_config_value(ckpt_config, "stop_file_path", None)
        self.load_model_only_folder = get_config_value(ckpt_config, "load_model_only_folder", None)
        self.load_optimizer = get_config_value(ckpt_config, "load_optimizer", True)
        if self.save_ckpt_folder:
            self.snapshot_ckpt_folder = get_config_value(
                ckpt_config, "snapshot_ckpt_folder", os.path.join(self.save_ckpt_folder, "snapshot")
            )
            self.async_upload_tmp_folder = get_config_value(
                ckpt_config, "async_upload_tmp_folder", "/dev/shm/internlm_tmp_ckpt/"
            )
        else:
            self.snapshot_ckpt_folder = None
            self.async_upload_tmp_folder = None

        self.async_upload = get_config_value(ckpt_config, "async_upload", False)

        # initialization storage manager
        init_storage_manager(self.enable_save_ckpt, self.async_upload_tmp_folder, self.async_upload)

        self.feishu_address = feishu_address
        self.storage_manager = get_storage_manager()
        self.snapshot_counter = 0

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dl = train_dl
        self.model_config = model_config
        self.model_config_file = model_config_file

        if self.stop_file_path and gpc.get_global_rank() == 0:
            dir_path = os.path.dirname(self.stop_file_path)
            if dir_path != "" and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(self.stop_file_path, "w", encoding="utf-8") as f:
                f.write("0")

        # Compatible with old interfaces.
        self.auto_resume = get_config_value(ckpt_config, "auto_resume_latest_ckpt", True)
        if "load_given_ckpt" in ckpt_config:
            self.auto_resume = not ckpt_config.load_given_ckpt

        self.load_ckpt_info = get_config_value(ckpt_config, "load_ckpt_info", None)
        if self.load_ckpt_info is None:
            load_ckpt_folder = get_config_value(ckpt_config, "load_ckpt_folder", None)
            if isinstance(load_ckpt_folder, str):
                self.load_ckpt_info = dict(
                    path=load_ckpt_folder, content=("model", "sampler", "optimizer"), ckpt_type="normal"
                )

        if self.load_model_only_folder is not None:
            assert (
                self.load_ckpt_info is None
            ), "Detect 'load_ckpt_folder' and 'load_model_only_folder' set at the same time, \
# and 'load_given_ckpt' is True, so internlm will load from 'load_ckpt_folder'"
            self.load_ckpt_info = dict(path=self.load_model_only_folder, content=["model"], ckpt_type="normal")

        if self.auto_resume:
            self.load_ckpt_info = self.query_lastest_ckpt()

        if self.stop_file_path is None:
            if gpc.is_rank_for_log():
                logger.warning("no set stop_file_path, quit_signal_handler is disable")

        # set load mask.
        if self.load_ckpt_info:
            if gpc.is_rank_for_log():
                logger.info(f"load_ckpt_info : {self.load_ckpt_info}")
            assert (
                "path" in self.load_ckpt_info
                and "content" in self.load_ckpt_info
                and "ckpt_type" in self.load_ckpt_info
            ), "please set content in ckpt setting, eg: ckpt = dict(path='', content=['model'], ckpt_type='normal')"
            assert set(self.load_ckpt_info["content"]).issubset(
                set(["model", "sampler", "optimizer", "scheduler", "all"])
            )
            assert isinstance(self.load_ckpt_info["content"], tuple)

            if "content" not in self.load_ckpt_info:
                self.load_ckpt_info["content"] = CheckpointLoadMask.LOAD_ALL
            else:
                load_mask = 0b0000
                for load_ctx in self.load_ckpt_info["content"]:
                    if load_ctx.lower() == "model":
                        load_mask |= int(CheckpointLoadMask.LOAD_MODEL)
                    if load_ctx.lower() == "optimizer":
                        load_mask |= CheckpointLoadMask.LOAD_OPTIMIZER
                    if load_ctx.lower() == "scheduler":
                        load_mask |= CheckpointLoadMask.LOAD_SCHEDULER
                    if load_ctx.lower() == "sampler":
                        load_mask |= CheckpointLoadMask.LOAD_SAMPLER
                    if load_ctx.lower() == "all":
                        load_mask |= CheckpointLoadMask.LOAD_ALL

                self.load_ckpt_info["content"] = load_mask

            assert set([self.load_ckpt_info["ckpt_type"]]).issubset(set(["normal"]))
            if self.load_ckpt_info["ckpt_type"] == "normal":
                self.load_ckpt_info["ckpt_type"] = CheckpointLoadType.INTERNLM

        # test storage setting is ok.
        if self.enable_save_ckpt:
            self.try_ping_storage()

    def quit_signal_handler(self, train_state) -> bool:
        """
        Exit signal detection function, if we write the exit step in the 'QUIT_FILE_PATH' file,
        all ranks will save ckpt and exit.
        Negative integer step means save ckpt.
        Positive integer step means save ckpt and quit.

        Args:
            train_state (TrainState):
        Returns:
            bool: whether to quit.
        """
        now_break, now_save_ckpt, save_type = False, False, CheckpointSaveType.NORMAL_CHECKPOINT

        if self.stop_file_path is None:
            return now_break, now_save_ckpt, save_type

        with open(self.stop_file_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            msg = f.read()
            fcntl.flock(f, fcntl.LOCK_UN)
            action_step = int(msg)

        if action_step < 0 and abs(action_step) == train_state.step_count:
            now_save_ckpt = True

        if action_step > 0 and action_step == train_state.step_count:
            now_break, now_save_ckpt = True, True

        if action_step != 0 and gpc.is_rank_for_log():
            msg = "Stop" if action_step > 0 else "Save"
            action_step = abs(action_step)
            if train_state.step_count <= action_step:
                if self.feishu_address:
                    send_alert_message(
                        address=self.feishu_address,
                        message=f"training will {msg} at step_count {action_step}!\
now step_count is {train_state.step_count}",
                    )

        return now_break, now_save_ckpt, save_type

    def is_now_to_save_ckpt(self, train_state) -> (bool, CheckpointSaveType, bool):
        save_ckpts, save_type, now_break = False, CheckpointSaveType.NORMAL_CHECKPOINT, False
        if self.oss_snapshot_freq > 1 and train_state.step_count % self.oss_snapshot_freq == 0:
            save_ckpts, save_type = True, CheckpointSaveType.SNAPSHOT_CHECKPOINT
        if train_state.step_count % self.checkpoint_every == 0:
            save_ckpts, save_type = True, CheckpointSaveType.NORMAL_CHECKPOINT
        now_break, singal_save_ckpts, singal_save_type = self.quit_signal_handler(train_state)
        if save_ckpts is False:
            save_ckpts = singal_save_ckpts
            save_type = singal_save_type

        return save_ckpts, save_type, now_break

    def try_save_checkpoint(self, train_state):
        if not self.enable_save_ckpt:
            return False

        save_ckpts, save_type, now_break = self.is_now_to_save_ckpt(train_state)

        if save_ckpts:
            # Wait for the previous round of asynchronous upload storage to complete.
            self.storage_manager.wait()
            if save_type == CheckpointSaveType.SNAPSHOT_CHECKPOINT:
                # Snapshot number, with only two snapshots written alternately.
                self.snapshot_counter = (self.snapshot_counter + 1) % 2
                save_ckpt_folder = os.path.join(self.snapshot_ckpt_folder, f"{self.snapshot_counter}")
            else:
                save_ckpt_folder = os.path.join(self.save_ckpt_folder, str(train_state.step_count))

            self.save_checkpoint(
                folder=save_ckpt_folder,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                train_state=train_state,
                model_config=self.model_config,
                model_config_file=self.model_config_file,
            )

        return now_break

    def wait_async_upload_finish(self):
        """wait for all checkpoint uploads to be completed"""
        self.storage_manager.wait()
        torch.distributed.barrier()

    def query_latest_snapshot_step_boto3(self):
        """query_latest_snapshot_step_boto3
        Returns:
            Tuple(str, int): path of latest ckpt and ckpt step, if not found, None will return.
        """
        ckpt_list = self.storage_manager.get_fns(self.save_ckpt_folder)
        if ckpt_list is None or len(ckpt_list) == 0:
            return None, None

        max_normal_step = 0
        ckpt_list = list(map(lambda a: int(a.strip("/")) if a.strip("/").isdigit() else 0, ckpt_list))
        ckpt_list.sort(reverse=True)
        for ckpt in ckpt_list:
            fns_list = self.storage_manager.get_fns(os.path.join(self.save_ckpt_folder, str(ckpt)))
            for fn in fns_list:
                if fn.endswith(".step"):
                    max_normal_step = ckpt
                    break
            if max_normal_step != 0:
                break

        max_normal_step = ckpt_list[0]
        load_normal_ckpt_path = os.path.join(self.save_ckpt_folder, str(max_normal_step))

        snapshot_path_0 = os.path.join(self.save_ckpt_folder, "snapshot", "0")
        snapshot_path_1 = os.path.join(self.save_ckpt_folder, "snapshot", "1")
        ckpt_list_1 = self.storage_manager.get_fns(snapshot_path_0)
        ckpt_list_2 = self.storage_manager.get_fns(snapshot_path_1)
        max_step_0, max_step_1 = 0, 0
        if ckpt_list_1:
            for ckpt in ckpt_list_1:
                ckpt = ckpt.strip("/")
                if ckpt.endswith(".step"):
                    max_step_0 = max(max_step_0, int(ckpt.split(".")[0]))
        if ckpt_list_2:
            for ckpt in ckpt_list_2:
                ckpt = ckpt.strip("/")
                if ckpt.endswith(".step"):
                    max_step_1 = max(max_step_1, int(ckpt.split(".")[0]))

        snap_load_path = snapshot_path_0 if max_step_0 > max_step_1 else snapshot_path_1
        snap_step = max(max_step_0, max_step_1)
        load_path = snap_load_path if snap_step > max_normal_step else load_normal_ckpt_path
        load_step = max(snap_step, max_normal_step)
        return load_path, load_step

    def query_latest_snapshot_step_local(self):
        max_step, max_step_path = 0, None
        save_ckpt_folder = self.save_ckpt_folder.split(":")[1]
        for root, _, files in os.walk(save_ckpt_folder, followlinks=True):
            for fn in files:
                fn = fn.strip("/")
                if fn.endswith(".step"):
                    # We assume that both normal ckpt and snapshot ckpt will store the '.step' file
                    # as an integrity flag.
                    step = int(fn.rsplit(".", maxsplit=1)[0])
                    if max_step < step:
                        max_step = step
                        max_step_path = root

        return max_step_path, max_step

    def query_lastest_ckpt(self):
        latest_ckpt, step = None, -1
        # Training was automatically restarted by the process, forcing the latest snapshot to be read.
        if self.save_ckpt_folder:
            backend, _ = try_get_storage_backend(self.save_ckpt_folder)
            if backend == "boto3":
                latest_ckpt, step = self.query_latest_snapshot_step_boto3()
                if latest_ckpt and not latest_ckpt.startswith("boto3:"):
                    latest_ckpt = ":".join(["boto3", latest_ckpt])
            elif backend == "local":
                latest_ckpt, step = self.query_latest_snapshot_step_local()
                if latest_ckpt and not latest_ckpt.startswith("local:"):
                    latest_ckpt = ":".join(["local", latest_ckpt])

        if gpc.is_rank_for_log():
            logger.info(f"Found latest ckpt {latest_ckpt if latest_ckpt else 'None'}, step: {step}...")

        # TODO: add more check point type.
        return dict(path=latest_ckpt, content=("model", "sampler", "optimizer"), ckpt_type="normal")

    def try_resume_training(self, train_state: TrainState, current_time=""):
        if self.load_ckpt_info is None or self.load_ckpt_info["path"] is None:
            if gpc.is_rank_for_log():
                logger.info(
                    f"===========New Run {current_time} on host:{socket.gethostname()},rank={gpc.get_global_rank()},"
                    f"tp={gpc.get_local_rank(ParallelMode.TENSOR)},pp={gpc.get_local_rank(ParallelMode.PIPELINE)},"
                    f"dp={gpc.get_local_rank(ParallelMode.DATA)}==========="
                )
        else:
            if gpc.is_rank_for_log():
                logger.info(f"Try load_ckpt_folder: {self.load_ckpt_info}")

            if self.load_ckpt_info["ckpt_type"] == CheckpointLoadType.INTERNLM:
                load_content_str = self.try_load_normal_ckpt(self.load_ckpt_info, train_state)
                if gpc.is_rank_for_log():
                    logger.info(
                        f"===========Resume training from `{self.load_ckpt_info['path']}` {current_time} on host:"
                        f"{socket.gethostname()}==========="
                    )
                    logger.info(f"===========Load contents are: {load_content_str}")
            else:
                raise NotImplementedError(f"unsupport ckpt_type : {self.load_ckpt_info['ckpt_type']}")

    def try_load_normal_ckpt(self, load_info, train_state: TrainState):
        load_content_str = ""
        load_ckpt_folder = load_info["path"]
        load_content = load_info["content"]

        if load_content & CheckpointLoadMask.LOAD_MODEL:
            load_model_checkpoint(folder=load_ckpt_folder, model=self.model)
            load_content_str += "model weigths, "

        if load_content | CheckpointLoadMask.LOAD_MODEL != CheckpointLoadMask.LOAD_MODEL:
            # load training states.
            load_context(load_ckpt_folder, train_state)

            # load optimzier states.
            if load_content & CheckpointLoadMask.LOAD_OPTIMIZER:
                load_optimizer_checkpoint(load_ckpt_folder, self.optimizer)
                load_content_str += "optimizer, "
            else:
                if gpc.is_rank_for_log():
                    logger.warning("CheckpointManager has no 'optimizer', skip reload optim checkpoint!")
                # If we use AMP optimizer, we need to update its fp32 buffer as newly loaded weights value.
                # Or we must ensure that loading model weights must be done before zero is initialized.
                if isinstance(self.optimizer, HybridZeroOptimizer):
                    for group_id, param_group in enumerate(self.optimizer.optim.param_groups):
                        if self.param_group_has_params[group_id]:
                            # flatten fp16 params have already been updated by 'load_model_checkpoint'
                            fp16_flat_current_rank = self.optimizer._param_store.get_flat_fp16_param_by_rank_group(
                                self._zero_local_rank, group_id
                            )
                            # param_group["params"] is fp32 flatten optimizer states of this zero rank.
                            param_group["params"][0].copy_(fp16_flat_current_rank.float())

            # load lr scheduler states.
            if load_content & CheckpointLoadMask.LOAD_SCHEDULER:
                if self.lr_scheduler:
                    load_scheduler(load_ckpt_folder, self.lr_scheduler, self.optimizer, train_state)
                    load_content_str += "lr_scheduler, "
                else:
                    if gpc.is_rank_for_log():
                        logger.warning("CheckpointManager has no 'lr_scheduler', skip reload lr_scheduler checkpoint!")

            # load dataloader sampler states.
            if load_content & CheckpointLoadMask.LOAD_SAMPLER:
                if hasattr(train_state, "batch_sampler") and not isinstance(
                    train_state.batch_sampler, torch.utils.data.sampler.BatchSampler
                ):
                    load_sampler(load_ckpt_folder, self.train_dl.batch_sampler)
                    # track the actual updates of sampler when using weighted sampling
                    train_state.init_batch_sampler(self.train_dl.batch_sampler)
                    load_content_str += "batch_sampler, "
                else:
                    if gpc.is_rank_for_log():
                        logger.warning("CheckpointManager skip reload 'batch_sampler'")

                # reload data state dict.
                if hasattr(train_state, "data_state_dict"):
                    self.train_dl.dataset.load_state_dict(
                        llm_load(os.path.join(load_ckpt_folder, "sampler_0.pt")), ckpt_path=load_ckpt_folder
                    )
                    load_content_str += "data_state_dict, "
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(
                            "CheckpointManager has no 'data_state_dict', skip reload data_state_dict checkpoint!"
                        )
        return load_content_str

    def save_checkpoint(
        self,
        folder,
        model,
        optimizer,
        scheduler,
        train_state: TrainState,
        model_config: Dict = None,
        model_config_file: str = None,
    ):
        """
        Save checkpoint to the given folder path.
        """

        start = time.time()
        self.set_save_folder(folder, train_state.step_count)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if gpc.is_rank_for_log():
            logger.info(f"Saving checkpoint to `{folder}` at batch count:{train_state.step_count}...")

        timer("save-model").start()
        save_model_checkpoint(folder=folder, model=model)
        timer("save-model").stop()

        timer("save-optimizer").start()
        save_optimizer_checkpoint(optim=optimizer, state_path=folder)
        timer("save-optimizer").stop()

        if (
            hasattr(train_state, "data_state_dict")
            and gpc.get_local_rank(ParallelMode.TENSOR) == 0
            and gpc.get_local_rank(ParallelMode.PIPELINE) == 0
        ):
            llm_save(
                os.path.join(folder, f"sampler_{gpc.get_local_rank(ParallelMode.DATA)}.pt"),
                saved_obj=train_state.data_state_dict,
            )

        if gpc.is_rank_for_log():
            if scheduler:
                scheduler_states = scheduler.state_dict()
                llm_save(os.path.join(folder, "schedulder.pt"), saved_obj=scheduler_states)

            if hasattr(train_state, "batch_sampler") and not isinstance(
                train_state.batch_sampler, torch.utils.data.sampler.BatchSampler
            ):
                sampler_state = train_state.batch_sampler.state_dict()
                llm_save(os.path.join(folder, "sampler.pt"), saved_obj=sampler_state)
            llm_save(os.path.join(folder, "context.pt"), saved_obj=train_state.state_dict())

            if model_config is not None:
                # Model configuration dictionary.
                llm_save(os.path.join(folder, "model_config.pt"), saved_obj=model_config)

            if model_config_file is not None:
                # The complete training config file content, stored in binary format.
                llm_save(os.path.join(folder, "config_file.pt"), saved_obj=model_config_file)

        torch.distributed.barrier()

        if gpc.is_rank_for_log():
            timer.log(["save-model", "save-optimizer"], logger=logger)
            logger.info(f"Step: {train_state.step_count}, rank 0 save ckpt use {time.time() - start:.3f} s")
            if self.storage_manager.async_mode is False:
                llm_save(
                    os.path.join(folder, f"{train_state.step_count}.step"),
                    saved_obj=dict({"step": train_state.step_count}),
                )

    def set_save_folder(self, folder, step):
        self.storage_manager.latest_save_folder = folder
        self.storage_manager.latest_save_step = step

    def try_ping_storage(self):
        if gpc.get_global_rank() % 8 == 0:
            buff = torch.ones((1, 64, 64), dtype=torch.bfloat16)
            test_fn = os.path.join(self.save_ckpt_folder, f"pings/{socket.gethostname()}.ping")
            self.storage_manager.save(test_fn, buff)
            self.storage_manager.wait()
            self.storage_manager.load(test_fn)
            del buff
