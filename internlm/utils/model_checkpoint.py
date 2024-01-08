#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import inspect
import os
import re
import socket
import time
from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, Union

import torch
from torch.distributed._shard.api import load_with_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.trainer import TrainState
from internlm.initialize.launch import get_config_value
from internlm.initialize.legacy.launch import (
    auto_resume_sanity_check,
    ckpt_info_sanity_check,
)
from internlm.model.moe import MoE
from internlm.monitor import send_alert_message
from internlm.solver.optimizer import HybridZeroOptimizer, reload_zero_fp32_buff
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
from internlm.utils.timeout import llm_timeout

logger = get_logger(__file__)


class CheckpointSaveType(Enum):
    NORMAL_CHECKPOINT = 1
    SNAPSHOT_CHECKPOINT = 2


class CheckpointLoadType(Enum):
    INTERNLM = "internlm"
    HF_LLAMA = "hf_llama"
    LLAMA = "llama"


# The load method implemented by internlm by default does not use string representation types,
# but uses enumeration types defined in advance.
LOAD_TYPE_DICT = {
    "internlm": CheckpointLoadType.INTERNLM,
    "hf_llama": CheckpointLoadType.HF_LLAMA,
    "llama": CheckpointLoadType.LLAMA,
}


class CheckpointLoadContent:
    MODEL = "model"
    SAMPLER = "sampler"
    OPIMIZER = "optimizer"
    SCHEDULAER = "scheduler"


class CheckpointLoadMethod:
    """The registration class of the checkpoint loading method,
    users can define their own custom ckpt loading methods."""

    LOAD_FUNC_SIG = None
    LOAD_TYPE_FUNC = {}

    @staticmethod
    def convert_load_type(load_type: str) -> Union[CheckpointLoadType, str]:
        if load_type.lower() in LOAD_TYPE_DICT:
            # The ckpt load method implemented by internlm by default.
            return LOAD_TYPE_DICT[load_type.lower()]
        else:
            # If it is a user-defined field, we do not do any conversion and represent it as a string.
            return load_type

    @staticmethod
    def register_ckpt_load_type(load_type: Union[str, CheckpointLoadType], load_func: Callable):
        if load_type in CheckpointLoadMethod.LOAD_TYPE_FUNC and gpc.is_rank_for_log():
            logger.warning(f"{load_type} has already been registered!")
            return

        CheckpointLoadMethod.LOAD_TYPE_FUNC.update({load_type: load_func})

        if load_type in (
            CheckpointLoadType.INTERNLM,
            CheckpointLoadType.HF_LLAMA,
            CheckpointLoadType.LLAMA,
        ):
            CheckpointLoadMethod.LOAD_FUNC_SIG = inspect.signature(load_func)
        else:
            if inspect.signature(load_func) != CheckpointLoadMethod.LOAD_FUNC_SIG and gpc.is_rank_for_log():
                logger.warning(
                    f"The registered signature {inspect.signature(load_func)} of the loaded model is not same as: "
                    f"{CheckpointLoadMethod.LOAD_FUNC_SIG}"
                )

    @staticmethod
    def get_ckpt_load_type_func(load_type: Union[str, CheckpointLoadType]):
        return CheckpointLoadMethod.LOAD_TYPE_FUNC[load_type]


class CheckpointLoadMask:
    """
    According to the content field in the incoming ckpt_info, decide which components to load.
    """

    LOAD_CONTENT_DICT = {
        "model": CheckpointLoadContent.MODEL,
        "sampler": CheckpointLoadContent.SAMPLER,
        "optimizer": CheckpointLoadContent.OPIMIZER,
        "scheduler": CheckpointLoadContent.SCHEDULAER,
    }

    def __init__(self, content: tuple) -> None:
        self.load_set = set(map(lambda x: x.lower(), content))
        if "all" in self.load_set:
            self.load_set = set(CheckpointLoadMask.LOAD_CONTENT_DICT.values())
        else:
            self.load_set = set(map(lambda x: CheckpointLoadMask.LOAD_CONTENT_DICT[x.lower()], content))

    def need_load(self, content: CheckpointLoadContent):
        return content in self.load_set

    def not_only_load(self, content: CheckpointLoadContent):
        return content in self.load_set and len(self.load_set) > 1

    def only_load(self, content: CheckpointLoadContent):
        return set((content,)) == self.load_set

    def __str__(self) -> str:
        return f"{self.load_set}."

    def __repr__(self) -> str:
        return f"{self.load_set}."


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


def get_shard_state_dict(shard_model):
    """
    Only used for FSDP module saving.
    It's a warper of model.state_dict() and with the context of 'FSDP.state_dict_type', the sharded parameter
    (saved as model.flat_param_xx in sharded FSDP module) will be gathered at every gpu.
    'offload_to_cpu' means that the model states are to be offloaded to cpu chunk by chunk, avoiding OOM in gpu

    """

    # FSDP model can only save with sharded shape SHARDED_STATE_DICT when set use_orig_params=True
    with FSDP.state_dict_type(shard_model, StateDictType.SHARDED_STATE_DICT):
        shard_states = shard_model.state_dict()

    return shard_states


def load_shard_state_dict(shard_model, shard_state, **kwargs):
    """
    Only used for FSDP module loading.

    """

    with FSDP.state_dict_type(shard_model, StateDictType.SHARDED_STATE_DICT):
        missing_k, unexpected_keys = shard_model.load_state_dict(shard_state, kwargs)

    return (missing_k, unexpected_keys)


def process_load_info(load_info):
    load_content_str = ""
    load_ckpt_folder = load_info["path"]
    load_content: CheckpointLoadMask = load_info["content"]
    if gpc.is_rank_for_log():
        logger.info(f"Try load_ckpt_folder: {load_ckpt_folder}")

    return load_content_str, load_ckpt_folder, load_content


def try_load_LLAMA_ckpt(ckpt_mm, load_info, train_state: TrainState):  # pylint: disable=W0613
    load_content_str, load_ckpt_folder, load_content = process_load_info(load_info)
    if load_content.need_load(CheckpointLoadContent.MODEL):
        load_llama_pretrained_weights(folder=load_ckpt_folder, model=ckpt_mm.model)
        load_content_str += f"{CheckpointLoadContent.MODEL}, "


def try_load_hf_LLAMA_ckpt(ckpt_mm, load_info, train_state: TrainState):  # pylint: disable=W0613
    load_content_str, load_ckpt_folder, load_content = process_load_info(load_info)
    if load_content.need_load(CheckpointLoadContent.MODEL):
        load_hf_llama_pretrained_weights(folder=load_ckpt_folder, model=ckpt_mm.model)
        load_content_str += f"{CheckpointLoadContent.MODEL}, "


def try_load_internlm_ckpt(ckpt_mm, load_info, train_state: TrainState):
    load_content_str, load_ckpt_folder, load_content = process_load_info(load_info)

    if load_content.need_load(CheckpointLoadContent.MODEL):
        load_model_checkpoint(folder=load_ckpt_folder, model=ckpt_mm.model)
        load_content_str += f"{CheckpointLoadContent.MODEL}, "

    if load_content.not_only_load(CheckpointLoadContent.MODEL):
        # load training states.
        load_context(load_ckpt_folder, train_state)

        # load optimizer states.
        if load_content.need_load(CheckpointLoadContent.OPIMIZER):
            load_optimizer_checkpoint(load_ckpt_folder, ckpt_mm.optimizer)
            load_content_str += f"{CheckpointLoadContent.OPIMIZER}, "
        else:
            if gpc.is_rank_for_log():
                logger.warning("CheckpointManager has no 'optimizer', skip reload optim checkpoint!")

        # load lr scheduler states.
        if load_content.need_load(CheckpointLoadContent.SCHEDULAER):
            if ckpt_mm.lr_scheduler:
                load_scheduler(load_ckpt_folder, ckpt_mm.lr_scheduler, ckpt_mm.optimizer, train_state)
                load_content_str += f"{CheckpointLoadContent.SCHEDULAER}, "
            else:
                if gpc.is_rank_for_log():
                    logger.warning("CheckpointManager has no 'lr_scheduler', skip reload lr_scheduler checkpoint!")

            if not load_content.need_load(CheckpointLoadContent.OPIMIZER):
                if ckpt_mm.lr_scheduler and train_state:
                    gpc.config.only_load_lr = True
                    load_optimizer_checkpoint(load_ckpt_folder, ckpt_mm.optimizer)
                    gpc.config.only_load_lr = False

        # load dataloader sampler states.
        if load_content.need_load(CheckpointLoadContent.SAMPLER):
            if hasattr(train_state, "batch_sampler") and not isinstance(
                train_state.batch_sampler, torch.utils.data.sampler.BatchSampler
            ):
                load_sampler(load_ckpt_folder, ckpt_mm.train_dl.batch_sampler)
                # track the actual updates of sampler when using weighted sampling
                train_state.init_batch_sampler(ckpt_mm.train_dl.batch_sampler)
                load_content_str += f"{CheckpointLoadContent.SAMPLER}, "
            else:
                if gpc.is_rank_for_log():
                    logger.warning("CheckpointManager skip reload 'batch_sampler'")

            # reload data state dict.
            if hasattr(train_state, "data_state_dict"):
                ckpt_mm.train_dl.dataset.load_state_dict(
                    llm_load(os.path.join(load_ckpt_folder, "sampler_0.pt")), ckpt_path=load_ckpt_folder
                )
                load_content_str += f"{CheckpointLoadContent.SAMPLER}, "
            else:
                if gpc.is_rank_for_log():
                    logger.warning(
                        "CheckpointManager has no 'data_state_dict', skip reload data_state_dict checkpoint!"
                    )
    return load_content_str


def save_model_checkpoint(folder, model):
    """
    Save the model according to the relationship between tp and dp. The principle is that the data of each tp
    will not be gathered and saved separately, which is equivalent to actual sharding. The saved weight is named
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If fsdp is activated, the saved weight is named:
    - folder
        - model_tp{tp_rank}_pp{pp_rank}_zo{zo_rank}

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.

    Args:
        folder: The folder to save the model
        model: The model to be saved
    """

    if gpc.config.parallel.zero1.fsdp:
        states = get_shard_state_dict(model)
    else:
        states = model.state_dict()

    # get non-expert parameters
    states = get_non_moe_state_dict(states)
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
            if gpc.config.parallel.zero1.fsdp:
                for j in range(dp_size):
                    should_save_rank_pair.add((i, j))
            else:
                should_save_rank_pair.add((i, i % dp_size))

            if (tp_rank, dp_rank) in should_save_rank_pair:
                f_dp = f"_dp{dp_rank}" if gpc.config.parallel.zero1.fsdp else ""
                fn = f"model_tp{tp_rank}_pp{pp_rank}{f_dp}.pt"
                fp = os.path.join(folder, fn)
                llm_save(fp, saved_obj=states)
                if not gpc.config.parallel.zero1.fsdp or dp_rank == tp_rank % dp_size:
                    topo_fn = f"topo_tp{tp_rank}_pp{pp_rank}.json"
                    topo_fp = os.path.join(folder, topo_fn)
                    llm_save(topo_fp, saved_obj=topo)

        # try to save expert parameter to separate files if model have moe layer
        expert_dp_size = gpc.get_world_size(ParallelMode.EXPERT_DATA)
        expert_dp_rank = gpc.get_local_rank(ParallelMode.EXPERT_DATA)
        should_save_rank_pair.clear()
        for i in range(tp_size):
            should_save_rank_pair.add((i, i % expert_dp_size))

        if (tp_rank, expert_dp_rank) in should_save_rank_pair:
            try_save_moe_checkpoint(folder, model, tp_rank, pp_rank)

    torch.distributed.barrier()


def load_llama_pretrained_weights(folder, model):
    model = model.model
    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = get_fns(folder)
    model_fns = [os.path.join(folder, fn) for fn in fns if fn.endswith(".pth") or fn.endswith(".pt")]
    model_fns.sort()

    old_tp = len(model_fns)
    cur_tp = gpc.get_world_size(ParallelMode.TENSOR)
    # If the two tp are inconsistent, you need to consider the merge before splitting
    if old_tp != cur_tp:
        raise RuntimeError(
            f"Your current tp is `{cur_tp}`, but the tp in folder:`{folder}` is `{old_tp}`, use `` to convert first"
        )

    states = llm_load(model_fns[gpc.get_local_rank(ParallelMode.TENSOR)], map_location="cpu")

    current_states = {}
    for idx, i in enumerate(range(model.first_layer, model.last_layer)):
        if gpc.config.model_type == "LLAMA2":
            # LLAMA's w2 and w3 are in reverse order
            w2 = states.pop(f"layers.{i}.feed_forward.w2.weight")
            w3 = states.pop(f"layers.{i}.feed_forward.w3.weight")
            states[f"layers.{i}.feed_forward.w2.weight"] = w3
            states[f"layers.{i}.feed_forward.w3.weight"] = w2
        if "rope.freqs" in states:
            states[f"layers.{i}.attention.rotary_emb.inv_freq"] = states["rope.freqs"]
        for name in list(states.keys()):
            if f".{i}." in name:
                current_states[name.replace(f".{i}.", f".{idx}.")] = states.pop(name)

    model_state_keys = set(list(model.state_dict().keys()))

    if "tok_embeddings.weight" in model_state_keys:
        current_states["tok_embeddings.weight"] = states["tok_embeddings.weight"]
        assert model.first_layer == 0, f"Expect model.NaiveAMPModel to be 0, but got {model.first_layer}"
    if "output.weight" in model_state_keys:
        current_states["norm.weight"] = states["norm.weight"]
        current_states["output.weight"] = states["output.weight"]
    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
        )

    del states
    del current_states
    torch.cuda.empty_cache()


def load_hf_llama_pretrained_weights(folder, model):
    model = model.model
    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = get_fns(folder)
    model_fns = [os.path.join(folder, fn) for fn in fns if fn.endswith(".bin") and fn.startswith("pytorch_model")]
    model_fns.sort()

    states = {}

    for model_fn in model_fns:
        states.update(llm_load(model_fn, map_location="cpu"))

    deep_split = getattr(model, "deep_split", False)
    if deep_split:
        print("using deep split when loading pretrained weights!")

    current_states = {}
    for idx, i in enumerate(range(model.first_layer, model.last_layer)):
        if gpc.config.model_type == "LLAMA2":
            if deep_split:
                layer_ids = i // 2
            else:
                layer_ids = i

            if not deep_split or (i + 2) % 2 == 0:
                states[f"layers.{i}.attention.wq.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.q_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wk.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.k_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wv.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.v_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wo.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.o_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=1,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention_norm.weight"] = states.pop(
                    f"model.layers.{layer_ids}.input_layernorm.weight"
                )

            if not deep_split or (i + 2) % 2 == 1:
                states[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.gate_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.up_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.down_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=1,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]

                states[f"layers.{i}.ffn_norm.weight"] = states.pop(
                    f"model.layers.{layer_ids}.post_attention_layernorm.weight"
                )

            if f"model.layers.{layer_ids}.self_attn.rotary_emb.inv_freq" in states:
                states.pop(f"model.layers.{layer_ids}.self_attn.rotary_emb.inv_freq")
        for name in list(states.keys()):
            if name.startswith(f"layers.{i}"):
                current_states[name.replace(f".{i}.", f".{idx}.")] = states.pop(name)

    model_state_keys = set(list(model.state_dict().keys()))

    if "tok_embeddings.weight" in model_state_keys or "tok_embeddings.word_embeddings.weight" in model_state_keys:
        if gpc.config.model.get("embed_split_hidden", True):
            current_states["tok_embeddings.weight"] = torch.chunk(
                states["model.embed_tokens.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=1
            )[gpc.get_local_rank(ParallelMode.TENSOR)]
        else:
            current_states["tok_embeddings.word_embeddings.weight"] = torch.chunk(
                states["model.embed_tokens.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=1
            )[gpc.get_local_rank(ParallelMode.TENSOR)]
        assert model.first_layer == 0, f"Expect model.first_layer to be 0, but got {model.first_layer}"
    if "output.weight" in model_state_keys:
        current_states["norm.weight"] = states["model.norm.weight"]
        current_states["output.weight"] = torch.chunk(
            states["lm_head.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=0
        )[gpc.get_local_rank(ParallelMode.TENSOR)]

    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
        )
    torch.cuda.empty_cache()


def load_model_checkpoint(folder, model):
    """
    There should be weights with names similar to the following under the folder.
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If fsdp is activated, the saved weight is named:
    - folder
        - model_tp{tp_rank}_pp{pp_rank}_zo{zo_rank}

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.
    """

    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
    dp_size = gpc.get_world_size(ParallelMode.DATA)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)

    fns = get_fns(folder)

    # avoid ckpt misuse between FSDP and no-FSDP
    test_fn = list([f for f in fns if f.startswith("model_t") and not f.endswith(".md5")]).pop()
    assert ("_dp" in test_fn and gpc.config.parallel.zero1.fsdp) or (
        "_dp" not in test_fn and not gpc.config.parallel.zero1.fsdp
    ), "FSDP model wants to load no-FSDP ckpts or reverse"

    max_pp, max_tp, max_zo = 0, 0, 0
    for fn in fns:
        if fn.startswith("model_t") and not fn.endswith(".md5"):
            segements = os.path.splitext(fn)[0].split("_")
            if gpc.config.parallel.zero1.fsdp:
                max_zo = max(max_zo, int(segements[-1][2:]))
                max_pp = max(max_pp, int(segements[-2][2:]))
                max_tp = max(max_tp, int(segements[-3][2:]))
            else:
                max_pp = max(max_pp, int(segements[-1][2:]))
                max_tp = max(max_tp, int(segements[-2][2:]))

    assert (
        pp_size == max_pp + 1
    ), f"The weights are save for {max_pp+1} pipelines, while current has {pp_size} pipelines"
    assert (
        tp_size == max_tp + 1
    ), f"The weights are save for {max_tp+1} parallelism, while current has {tp_size} tensor parallelism"
    if gpc.config.parallel.zero1.fsdp:
        assert (
            dp_size == max_zo + 1
        ), f"The weights are save for {max_zo+1} FSDP shards , while current has {dp_size} FSDP shards"

    if gpc.config.parallel.zero1.fsdp:
        should_load_name = f"model_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}.pt"
    else:
        should_load_name = f"model_tp{tp_rank}_pp{pp_rank}.pt"
    fp = os.path.join(folder, should_load_name)

    # for FSDP shards loading, we need to set process group
    with load_with_process_group(gpc.get_group(ParallelMode.ZERO1)):
        states = llm_load(fp, map_location=get_current_device())

    """
    # need convert the gate parameters to float32 (to fit deepspeed style mechanism), it may cause round-off in
    # gate.weight. The conversion will also be done when doing forward. so we can just comment it out. this make
    # the gate parameters to be float16 before forward.
    for key in list(states.keys()):
        if 'moe_layer.gate.wg.weight' in key:
            states[key] = states[key].float()
            print("load: ", states[key].float(),flush=True)
    """

    # try to load expert parameter to separate files if model have moe layer
    try_load_moe_checkpoint(folder, model, states, tp_rank, pp_rank)

    if gpc.config.parallel.zero1.fsdp:
        missing_k, unexpected_keys = load_shard_state_dict(model, states, strict=False)
    else:
        missing_k, unexpected_keys = model.load_state_dict(states, strict=False)
    if len(missing_k) != 0:
        logger.warning(f"Warning: missing keys {missing_k}")
    if len(unexpected_keys) != 0:
        logger.warning(f"Warning: unexpected keys {unexpected_keys}")

    # avoid to cuda oom, Ref: https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/11
    del states
    torch.cuda.empty_cache()


def try_save_moe_checkpoint(folder, model, tp_rank, pp_rank):
    # Using layer_#_expert_# to save the model's expert state_dict，a hack.
    pipeline_stage_size = gpc.config.model.num_layers // gpc.get_world_size(ParallelMode.PIPELINE)
    moe_layer_id = pp_rank * pipeline_stage_size
    for n_module, module in model.named_modules():
        if isinstance(module, MoE):  # and deepspeed.comm.get_rank() == 0:
            num_local_experts = module.num_local_experts
            expp_rank = gpc.get_local_rank(ParallelMode.EXPERT)

            # get all moe parameters
            moe_state_dict = {}
            for n, p in module.state_dict().items():
                if "expert" in n and "moe_layer.gate" not in n:
                    moe_state_dict[n_module + "." + n] = p
            moe_str_prefix = ".moe_layer.experts.wrapped_experts."
            # Reorder the moe name rank, so that each checkpoint only has one expert
            experts_state_dict = defaultdict(dict)
            for key in list(moe_state_dict.keys()):
                m = re.match(f".*{moe_str_prefix}([0-9]+).*", key)

                local_expert_id = None
                if not m:
                    logger.warning(f"No expert found in key {key}.")
                else:
                    local_expert_id = m.group(1)

                global_expert_id = expp_rank * num_local_experts + int(local_expert_id)
                expert_key = key.replace(f"{moe_str_prefix}{local_expert_id}", f"{moe_str_prefix}{global_expert_id}")

                # truncating extra tensor (shared) storage
                truncated = moe_state_dict.pop(key).clone().detach()
                experts_state_dict[str(global_expert_id)][expert_key] = truncated

            # let save the moe parameters
            for global_expert_id, expert_state_dict in experts_state_dict.items():
                # save the moe parameters
                fn = f"model_moe_layer{moe_layer_id}_expert{global_expert_id}_tp{tp_rank}.pt"
                fp = os.path.join(folder, fn)
                llm_save(fp, saved_obj=expert_state_dict)
            moe_layer_id += 1


def get_non_moe_state_dict(full_state_dict):
    """
    Get the state dict of the non-moe layers
    """
    for key in list(full_state_dict.keys()):
        if "expert" in key and "moe_layer.gate" not in key:
            full_state_dict.pop(key)

    return full_state_dict


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
    zero_size = gpc.get_world_size(ParallelMode.ZERO1)
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
    fp = f"optimizer_tp{tp_rank}_pp{pp_rank}_zo{zero_rank}.pt"

    states = optim.state_dict()
    if isinstance(optim, HybridZeroOptimizer):
        if gpc.get_global_rank() < zero_size * tp_size * pp_size:
            llm_save(os.path.join(state_path, fp), states)
            if "zero_devide_optim_plan" in states:
                params_per_rank_id_dict = states.pop("zero_devide_optim_plan")
                fp_meta = os.path.join(state_path, optim.rank_unique_id)
                llm_save(fp_meta, params_per_rank_id_dict)
    else:
        llm_save(os.path.join(state_path, fp), states)


def try_load_moe_checkpoint(folder, model, state_dict, tp_rank, pp_rank):
    pipeline_stage_size = gpc.config.model.num_layers // gpc.get_world_size(ParallelMode.PIPELINE)
    moe_layer_id = pp_rank * pipeline_stage_size
    for _, module in model.named_modules():
        if isinstance(module, MoE):  # and deepspeed.comm.get_rank() == 0:
            num_local_experts = module.num_local_experts
            expp_rank = gpc.get_local_rank(ParallelMode.EXPERT)
            # loop all local_experts
            for local_expert_id in range(num_local_experts):
                global_expert_id = expp_rank * num_local_experts + local_expert_id
                fn = f"model_moe_layer{moe_layer_id}_expert{global_expert_id}_tp{tp_rank}.pt"
                fp = os.path.join(folder, fn)
                expert_state_dict = llm_load(fp, map_location=get_current_device())
                # Updating global -> local expert ids
                moe_str_prefix = ".moe_layer.experts.wrapped_experts."
                for key in list(expert_state_dict.keys()):
                    local_key = key.replace(f"{moe_str_prefix}{global_expert_id}", f"{moe_str_prefix}{local_expert_id}")
                    expert_state_dict[local_key] = expert_state_dict.pop(key)
                state_dict.update(expert_state_dict)
            moe_layer_id += 1


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
            if gpc.is_rank_for_log():
                logger.warning(
                    f"Read zero optimzer split file '{fp_meta}', for '{e}'"
                    f"Please check whether loading ckpts are saved with the HybridZeroOptimizer."
                )

    # compatible with old code that only have one param group, need to align with both parameter groups
    if len(states["base_optim_states"]["param_groups"]) == 1:
        for group in optim.param_groups:
            # for new added empty group, since it has no params, just create it fakely
            if len(group["params"]) == 0:
                states["base_optim_states"]["param_groups"].append(group)
            # for origin group, create new added attributes in recent updates
            else:
                saved_group = states["base_optim_states"]["param_groups"][0]
                saved_group["dp_mode"] = group["dp_mode"]
                saved_group["dtype"] = group["dtype"]

    optim.load_state_dict(states)
    del states
    torch.cuda.empty_cache()


def load_sampler(ckpt_path: str, sampler):
    sampler_states = llm_load(os.path.join(ckpt_path, "sampler.pt"))
    sampler.load_state_dict(sampler_states)
    if gpc.is_rank_for_log():
        pstate = copy.deepcopy(sampler_states)
        pstate.pop("indices", None)
        pstate.pop("rng_state", None)
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

    # compatible with old code that only have one param group
    if len(base_lrs) == 1:
        base_lrs = base_lrs * len(optimizer.param_groups)

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
            model (nn.module): model obj.
            optimizer (object): optimizer obj.
            lr_scheduler (object): lr_scheduler obj.
            model_config (dict): model config.
        """
        self.enable_save_ckpt = get_config_value(ckpt_config, "enable_save_ckpt", False)
        self.checkpoint_every = get_config_value(ckpt_config, "checkpoint_every", 100)
        self.save_ckpt_folder = get_config_value(ckpt_config, "save_ckpt_folder", None)
        self.oss_snapshot_freq: int = get_config_value(ckpt_config, "oss_snapshot_freq", 50)
        self.stop_file_path = get_config_value(ckpt_config, "stop_file_path", None)
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

        # Register defalut internlm ckpt load type.
        self.defalut_load_type_func = {
            CheckpointLoadType.INTERNLM: try_load_internlm_ckpt,
            CheckpointLoadType.HF_LLAMA: try_load_hf_LLAMA_ckpt,
            CheckpointLoadType.LLAMA: try_load_LLAMA_ckpt,
        }
        for ckpt_load_type in CheckpointLoadType:
            CheckpointLoadMethod.register_ckpt_load_type(ckpt_load_type, self.defalut_load_type_func[ckpt_load_type])

        # Init alter file.
        if self.stop_file_path and gpc.get_global_rank() == 0:
            dir_path = os.path.dirname(self.stop_file_path)
            if dir_path != "" and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(self.stop_file_path, "w", encoding="utf-8") as f:
                f.write("0")

        self.load_ckpt_info = get_config_value(ckpt_config, "load_ckpt_info", None)
        if self.load_ckpt_info is None:  # (legacy): Try Compatible with old interfaces
            self.load_ckpt_info = ckpt_info_sanity_check(ckpt_config)

        # Auto-reload latest checkpoint, it will overwrite the setting of 'load_ckpt_info'.
        self.auto_resume = get_config_value(ckpt_config, "auto_resume", None)
        if self.auto_resume is None:  # (legacy): Try Compatible with old interfaces
            self.auto_resume = auto_resume_sanity_check(ckpt_config)
        if self.auto_resume:
            self.load_ckpt_info = self.query_lastest_ckpt()

        if self.stop_file_path is None and gpc.is_rank_for_log():
            logger.warning("no set stop_file_path, quit_signal_handler is disable")

        # convert to internal representation
        if self.load_ckpt_info:
            assert (
                "path" in self.load_ckpt_info
                and "content" in self.load_ckpt_info
                and "ckpt_type" in self.load_ckpt_info
            ), "please set content in ckpt setting, eg: ckpt = dict(path='', content=['model'], ckpt_type='internlm')"

            if self.load_ckpt_info["content"] != ("model",):
                assert (
                    self.load_ckpt_info["ckpt_type"] == "internlm"
                ), "Only 'internlm' ckpt supports loading states other than 'model' !"

            # replace load_ckpt
            self.load_ckpt_info["content"] = CheckpointLoadMask(self.load_ckpt_info["content"])
            self.load_ckpt_info["ckpt_type"] = CheckpointLoadMethod.convert_load_type(self.load_ckpt_info["ckpt_type"])

        torch.distributed.barrier()
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

        with torch.no_grad():
            action_step_t = torch.zeros((1,), dtype=torch.int64).cuda()
            if gpc.get_global_rank() == 0:
                with open(self.stop_file_path, "r+", encoding="utf-8") as f:
                    f.seek(0)
                    msg = f.read()
                    action_step_t.fill_(int(msg))

            torch.distributed.broadcast(action_step_t, src=0)
            action_step = action_step_t.item()
            del action_step_t

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
        if train_state.step_count % self.checkpoint_every == 0 or train_state.step_count == train_state.total_steps:
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
        """Query the latest snapshot step from the storage backend.
        Currently, we only support the following storage backends: boto3, oss2 and volc.
        Returns:
            Tuple(str, int): path of latest ckpt and ckpt step, if not found, None will return.
        """
        ckpt_list = self.storage_manager.get_fns(self.save_ckpt_folder)
        if ckpt_list is None or len(ckpt_list) == 0:
            return None, None

        max_normal_step = 0
        # Return ckpt_list look like: ['pings', 'snapshot', '4']
        # Here we only try to find the ckpt folder named after step, ignoring snapshot and other folders.
        ckpt_list = [int(fn.strip("/")) for fn in ckpt_list if fn.strip("/").isdigit()]
        if len(ckpt_list) == 0:
            if gpc.is_rank_for_log():
                logger.warning("No available normal checkpoint found. Check your checkpoint path.")
        else:
            if gpc.is_rank_for_log():
                logger.info(f"Found available normal checkpoint: {ckpt_list}")

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
        ckpt_list_0 = self.storage_manager.get_fns(snapshot_path_0)
        ckpt_list_1 = self.storage_manager.get_fns(snapshot_path_1)

        def found_latest_snapshot(_ckpt_list):
            _max_step_snapshot = 0
            if _ckpt_list:
                for ckpt in _ckpt_list:
                    ckpt = ckpt.strip("/")
                    if ckpt.endswith(".step"):
                        _max_step_snapshot = max(_max_step_snapshot, int(ckpt.split(".")[0]))
            return _max_step_snapshot

        max_step_0 = found_latest_snapshot(ckpt_list_0)
        max_step_1 = found_latest_snapshot(ckpt_list_1)

        if sum([max_step_0, max_step_1, max_normal_step]) == 0:
            return None, None
        else:
            snap_load_path = snapshot_path_0 if max_step_0 > max_step_1 else snapshot_path_1
            snap_step = max(max_step_0, max_step_1)
            load_path = snap_load_path if snap_step > max_normal_step else load_normal_ckpt_path
            return load_path, max(snap_step, max_normal_step)

    def query_latest_snapshot_step_local(self):
        """Query the latest snapshot step from the local file system."""
        max_step, max_step_path = 0, None
        save_ckpt_folder = self.save_ckpt_folder.split(":")[1]
        for root, _, files in os.walk(save_ckpt_folder, followlinks=True):
            for fn in files:
                fn = fn.strip("/")
                if fn.endswith(".step"):
                    # We assume that both internlm ckpt and snapshot ckpt will store the '.step' file
                    # as an integrity flag.
                    step = int(fn.rsplit(".", maxsplit=1)[0])
                    if max_step < step:
                        max_step = step
                        max_step_path = root

        return max_step_path, max_step

    def query_lastest_ckpt(self):
        """Query the latest ckpt via the storage backend."""
        latest_ckpt, step = None, -1
        # Training was automatically restarted by the process, forcing the latest snapshot to be read.
        if self.save_ckpt_folder:
            backend, _ = try_get_storage_backend(self.save_ckpt_folder)
            if backend in ["boto3", "oss2", "volc"]:
                latest_ckpt, step = self.query_latest_snapshot_step_boto3()
            elif backend == "local":
                latest_ckpt, step = self.query_latest_snapshot_step_local()
            else:
                raise NotImplementedError(
                    f"Unsupported backend: {backend}, " "Currently only support `boto3`, `oss2`, `volc` and `local`"
                )

            if latest_ckpt and not latest_ckpt.startswith(backend + ":"):
                latest_ckpt = ":".join([backend, latest_ckpt])

        if gpc.is_rank_for_log():
            logger.info(f"Found latest ckpt {latest_ckpt if latest_ckpt else 'None'}, step: {step}...")

        return dict(path=latest_ckpt, content=("all",), ckpt_type="internlm")

    def try_resume_training(self, train_state: TrainState, current_time=""):
        if self.load_ckpt_info is None or self.load_ckpt_info["path"] is None:
            if gpc.is_rank_for_log():
                logger.info(
                    f"===========New Run {current_time} on host:{socket.gethostname()},rank={gpc.get_global_rank()},"
                    f"tp={gpc.get_local_rank(ParallelMode.TENSOR)},pp={gpc.get_local_rank(ParallelMode.PIPELINE)},"
                    f"dp={gpc.get_local_rank(ParallelMode.DATA)}==========="
                )
        else:
            load_path = self.load_ckpt_info["path"]
            load_content = self.load_ckpt_info["content"]
            load_type = self.load_ckpt_info["ckpt_type"]

            load_func = CheckpointLoadMethod.get_ckpt_load_type_func(load_type)
            load_content_str = load_func(self, self.load_ckpt_info, train_state)

            # If we only load model weight, we need rewrite zero optim's fp32 buffer.
            if load_content.only_load(CheckpointLoadContent.MODEL) and isinstance(self.optimizer, HybridZeroOptimizer):
                reload_zero_fp32_buff(self.optimizer)

            if gpc.is_rank_for_log():
                logger.info(f"load_ckpt_info : {self.load_ckpt_info}")
                logger.info(
                    f"===========Resume training from `{load_path}` {current_time} on host:"
                    f"{socket.gethostname()}==========="
                )
                if load_content_str:
                    logger.info(f"===========Load contents are: {load_content_str}")

    @llm_timeout(func_name="save_checkpoint")
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
        if gpc.is_rank_for_log():
            buff = torch.ones((1, 64, 64), dtype=torch.bfloat16)
            test_fn = os.path.join(self.save_ckpt_folder, f"pings/{socket.gethostname()}.ping")
            self.storage_manager.save(test_fn, buff)
            self.storage_manager.wait()
            self.storage_manager.load(test_fn)
            del buff
