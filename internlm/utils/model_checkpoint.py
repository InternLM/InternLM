#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import os
import time
from typing import Dict

import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.trainer import TrainState
from internlm.solver.optimizer import HybridZeroOptimizer
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.storage_manager import get_fns, llm_load, llm_save

logger = get_logger(__file__)


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


def save_checkpoint(folder, model, optimizer, scheduler, train_state: TrainState, model_config: Dict = None):
    """
    Save checkpoint to the given folder path.
    """

    start = time.time()
    torch.distributed.barrier()
    folder = os.path.join(folder, str(train_state.step_count))
    logger.info(
        f"Saving checkpoint to `{folder}` at batch count:{train_state.step_count} from rank:{gpc.get_global_rank()}..."
    )

    timer("save-model").start()
    save_model_checkpoint(folder=folder, model=model)
    timer("save-model").stop()

    timer("save-optimizer").start()
    save_optimizer_checkpoint(optim=optimizer, state_path=folder)
    timer("save-optimizer").stop()

    if gpc.is_rank_for_log():
        scheduler_states = scheduler.state_dict()
        llm_save(os.path.join(folder, "schedulder.pt"), saved_obj=scheduler_states)

        sampler_state = train_state.batch_sampler.state_dict()
        llm_save(os.path.join(folder, "sampler.pt"), saved_obj=sampler_state)
        llm_save(os.path.join(folder, "context.pt"), saved_obj=train_state.state_dict())

        if model_config is not None:
            llm_save(os.path.join(folder, "model_config.pt"), saved_obj=model_config)

    torch.distributed.barrier()

    if gpc.is_rank_for_log():
        timer.log(["save-model", "save-optimizer"], logger=logger)
        logger.info(f"Step: {train_state.step_count}, rank 0 save ckpt use {time.time() - start:.3f} s")


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


def load_context(ckpt_path: str, train_dl, train_state: TrainState):
    context_stuffs = llm_load(os.path.join(ckpt_path, "context.pt"))
    train_state.load_state_dict(context_stuffs, train_dl)
    if gpc.is_rank_for_log():
        logger.info(f"reload train_state:{train_state}")
    torch.cuda.empty_cache()


def load_scheduler(ckpt_path: str, lr_scheduler, optimizer, learning_rate, train_state: TrainState):
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
