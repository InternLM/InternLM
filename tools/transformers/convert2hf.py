# Copyright (c) InternLM. All rights reserved.
"""
python convert2hf.py --src /path/to/src --tgt /path/to/tgt \
       --max_shard 2G --max_pos 8192 \
       --tokenizer /path/to/tokenizer.model \
"""
import argparse
import gc
import json
import os
import re
import time

import torch
from internlm_model import InternLMConfig, InternLMForCausalLM, InternLMTokenizer
from tqdm import tqdm
from transformers.modeling_utils import no_init_weights

embedding_key_list = ["embedding.word_embeddings.weight", "embedding.weight", "tok_embeddings.weight", None]


def _find_max_tp_pp(names):
    ckpt_names = []
    for name in names:
        if name.startswith("model_t") and not name.endswith("md5"):
            # _t: avoid conflictint with model_config.pt
            ckpt_names.append(name)

    max_tp, max_pp = -1, -1
    for ckpt in ckpt_names:
        _, tp, pp = os.path.splitext(ckpt)[0].split("_")
        max_tp = max(max_tp, int(tp[2:]) + 1)
        max_pp = max(max_pp, int(pp[2:]) + 1)

    return max_tp, max_pp


def load_source(src):
    """
    load model_config.pt and model_tp{x}_pp{x}.pt from ``src``

    :return:
        - model_config: dict
        - states: 2-d array. states[i][j] stands for state_dict of tp_i pp_j
    """

    # config
    print("Config loading", flush=True)
    config_file = os.path.join(src, "model_config.pt")
    assert os.path.isfile(config_file), f"model_config.pt is not found in :{os.listdir(src)}"
    model_config = torch.load(config_file)
    print(model_config)
    print("Config loaded.", flush=True)

    # checkpoint
    # find tp pp
    assert os.path.isdir(src), "not a folder."
    ckpt_names = os.listdir(src)
    max_tp, max_pp = _find_max_tp_pp(ckpt_names)

    # 2-d array tp_rank, pp_rank
    print("Source Checkpoint Loading", flush=True)
    states = [[None for _ in range(max_pp)] for __ in range(max_tp)]
    for tp in tqdm(range(max_tp)):
        for pp in tqdm(range(max_pp)):
            ckpt_name = os.path.join(src, f"model_tp{tp}_pp{pp}.pt")
            states[tp][pp] = torch.load(ckpt_name, map_location="cpu")
    print("Source Checkpoint Loaded", flush=True)
    return model_config, states


def merge(states):
    """
    Merge state dicts of pipeline format and shift some layers.

    :return:
        - config: InternLMConfig
        - states: merged state dict
    """
    # merge pp
    merged_states = []
    print("Pipeline Merging", flush=True)
    for tp_state in tqdm(states):
        layer_shift = 0
        shifted_state = {}
        # shift key
        for tp_pp_state in tp_state:
            _layer_shift = 0
            keys = list(tp_pp_state.keys())
            for key in keys:
                if key.endswith(".inv_freq"):
                    continue
                match = re.search(r"\.\d+\.", key)
                name = key
                if match is not None:
                    # layers
                    s, e = match.span()
                    layer_idx = int(key[s + 1 : e - 1]) + layer_shift
                    _layer_shift = max(_layer_shift, int(key[s + 1 : e - 1]))
                    name = key[:s] + f".{layer_idx}." + key[e:]
                if name.startswith("model."):
                    name = name[6:]
                shifted_state[name] = tp_pp_state[key]
            layer_shift += _layer_shift + 1

        merged_states.append(shifted_state)

    print("Pipeline Merged", flush=True)

    return merged_states


def convert(src, tgt, tokenizer, dtype, max_shard_size, max_pos, rope_scaling):
    """
    Convert state_dict to hf format.

    1. Load and merge state dict
    2. Convert to huggingface
    3. Load tokneizer and save it with ``tokenizer.save_pretrained``
    4. Load state dict to the model
    5. Call ``model.save_pretrained`` to save checkpoints.
    """
    # load states
    model_config, src_states = load_source(src)
    states = merge(src_states)
    del src_states

    num_shards = len(states)
    print("Converting to huggingface format...", flush=True)

    n_heads = model_config["num_attention_heads"]
    dim = model_config["hidden_size"]
    # n_heads_per_shard = n_heads // num_shards
    # dims_per_head = dim // n_heads
    intermediate_size = None

    print("Start converting...", flush=True)
    state_dict = {}
    for layer_i in tqdm(range(model_config["num_layers"])):
        wqkvs = [
            states[tp].pop(f"blocks.{layer_i}.mixer.Wqkv.weight").reshape(3, n_heads // num_shards, -1, dim)
            for tp in range(num_shards)
        ]
        bqkvs = [
            states[tp].pop(f"blocks.{layer_i}.mixer.Wqkv.bias").reshape(3, n_heads // num_shards, -1)
            for tp in range(num_shards)
        ]
        state_dict.update(
            {
                f"model.layers.{layer_i}.input_layernorm.weight": states[0][f"blocks.{layer_i}.norm1.weight"].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": states[0][
                    f"blocks.{layer_i}.norm2.weight"
                ].clone(),
            }
        )
        state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = torch.cat(
            [wqkvs[i][0] for i in range(num_shards)],
            dim=0,
        ).reshape(dim, dim)
        state_dict[f"model.layers.{layer_i}.self_attn.q_proj.bias"] = torch.cat(
            [bqkvs[i][0] for i in range(num_shards)],
            dim=0,
        ).reshape(-1)
        state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = torch.cat(
            [wqkvs[i][1] for i in range(num_shards)],
            dim=0,
        ).reshape(dim, dim)
        state_dict[f"model.layers.{layer_i}.self_attn.k_proj.bias"] = torch.cat(
            [bqkvs[i][1] for i in range(num_shards)],
            dim=0,
        ).reshape(-1)
        state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
            [wqkvs[i][2] for i in range(num_shards)],
            dim=0,
        ).reshape(dim, dim)
        state_dict[f"model.layers.{layer_i}.self_attn.v_proj.bias"] = torch.cat(
            [bqkvs[i][2] for i in range(num_shards)],
            dim=0,
        ).reshape(-1)

        state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
            [states[i][f"blocks.{layer_i}.mixer.out_proj.weight"] for i in range(num_shards)], dim=1
        )
        state_dict[f"model.layers.{layer_i}.self_attn.o_proj.bias"] = states[0][f"blocks.{layer_i}.mixer.out_proj.bias"]
        state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
            [states[i][f"blocks.{layer_i}.mlp.w1.weight"] for i in range(num_shards)], dim=0
        )
        intermediate_size, _ = state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"].shape
        state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
            [states[i][f"blocks.{layer_i}.mlp.w3.weight"] for i in range(num_shards)], dim=1
        )
        state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
            [states[i][f"blocks.{layer_i}.mlp.w2.weight"] for i in range(num_shards)], dim=0
        )

    # embedding
    for embedding_key in embedding_key_list:
        if embedding_key in states[0]:
            break
    if embedding_key is None:
        raise KeyError("Cannot find embedding key!")
    if model_config["embed_split_hidden"]:
        embed_concat_dim = 1
        tok_emb_list = [states[i][embedding_key] for i in range(num_shards)]
    else:
        embed_concat_dim = 0
        _, size_1 = states[0][embedding_key].shape
        embdim_pertp = size_1 // num_shards
        tok_emb_list = [
            torch.concat(
                [
                    states[tp][embedding_key][:, embdim_pertp * local_rank : embdim_pertp * (local_rank + 1)]
                    for tp in range(num_shards)
                ],
                dim=0,
            )
            for local_rank in range(num_shards)
        ]
    state_dict.update(
        {
            "model.norm.weight": states[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(tok_emb_list, dim=embed_concat_dim),
            "lm_head.weight": torch.cat([states[i]["head.weight"] for i in range(num_shards)], dim=0),
        },
    )

    # initialize model
    # tokenizer
    tokenizer = InternLMTokenizer(tokenizer)
    # config
    config = InternLMConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        intermediate_size=intermediate_size,
        num_attention_heads=model_config["num_attention_heads"],
        num_hidden_layers=model_config["num_layers"],
        rms_norm_eps=model_config["layer_norm_epsilon"],
        bias=True,
        rope_theta=model_config.get("rope_base", 10000),
        rope_scaling=rope_scaling,
    )
    # tokenizer
    config.max_position_embeddings = max_pos
    # set bos eos pad to avoid improper generation
    # since model.generate will create attention_mask
    # according to pad_token_id and bos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    # model
    print("Initializing model...", flush=True)
    start = time.time()
    with no_init_weights():
        model = InternLMForCausalLM._from_config(config, torch_dtype=dtype)
    print(f"Initializing model takes {time.time() - start}s", flush=True)
    model.load_state_dict(state_dict)

    del states
    gc.collect()
    print(f"Saving model to {tgt}...", flush=True)
    tokenizer.save_pretrained(tgt)
    model.save_pretrained(tgt, max_shard_size=max_shard_size)

    # fix auto_map in config
    with open(os.path.join(tgt, "config.json")) as fp:
        config_dict = json.load(fp)
    config_dict["auto_map"]["AutoModel"] = "modeling_internlm.InternLMForCausalLM"
    with open(os.path.join(tgt, "config.json"), "w") as fp:
        json.dump(config_dict, fp, indent=2)


def convert_tokenizer(src, tgt):
    assert os.path.isfile(src)
    tokenizer = InternLMTokenizer(src)
    tokenizer.save_pretrained(tgt)


def get_rope_scaling(args):
    if args.rotary_type == "origin":
        return None
    elif args.rotary_type == "dynamic":
        return {"type": args.rotary_type, "factor": args.scaling_factor}
    else:
        raise NotImplementedError(f"Unknown rope type {args.rotary_type}")


def print_args(args):
    print("-------------- Arguments --------------")
    print(f"Source Path: {args.src}")
    print(f"Target Path: {args.tgt}")
    print(f"Dtype: {args.dtype}")
    print(f"Max Shard Size: {args.max_shard}")
    print(f"Max Position Embedding: {args.max_pos}")
    print(f"Tokenizer Path: {args.tokenizer}")
    print(f"Rotary Type: {args.rotary_type}")
    print(f"Scaling Factor: {args.scaling_factor}")
    print("---------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--src", type=str, default=None, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="Data type after converting")
    parser.add_argument("--max_shard", type=str, default="10GB", help="Max size of every sharded checkpoint.")
    parser.add_argument("--max_pos", type=int, default=4096, help="Max position embedding of model.")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer model.")
    # rope
    parser.add_argument("--rotary_type", type=str, default="origin", help="Rope type", choices=["origin", "dynamic"])
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Scaling factor of dynamic rope.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    dtype = getattr(torch, args.dtype)
    rope_scaling = get_rope_scaling(args)

    assert args.src is not None, "--src is needed!"
    assert args.tokenizer is not None, "--tokenizer is needed!"
    start = time.time()
    convert(args.src, args.tgt, args.tokenizer, dtype, args.max_shard, args.max_pos, rope_scaling)
    print(f"Converting model takes {time.time() - start}s totally", flush=True)
