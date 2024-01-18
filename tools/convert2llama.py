# Copyright (c) InternLM. All rights reserved.
import argparse
import os
from collections import defaultdict

import torch
from einops import rearrange
from tqdm import tqdm
from transformers import AutoConfig


def split_wqkv(qkv, num_groups, q_per_kv, head_dim):
    """Split wqkv into wq, wk, wv."""
    qkv = qkv.T
    qkv = rearrange(qkv, "o (g n i) -> o g n i", g=num_groups, n=q_per_kv + 2, i=head_dim)

    q = qkv[..., :q_per_kv, :]
    k = qkv[..., q_per_kv : q_per_kv + 1, :]
    v = qkv[..., q_per_kv + 1 : q_per_kv + 2, :]

    q = rearrange(q, "o g n i -> o (g n i)", g=num_groups, n=q_per_kv, i=head_dim)
    k = rearrange(k, "o g n i -> o (g n i)", g=num_groups, n=1, i=head_dim)
    v = rearrange(v, "o g n i -> o (g n i)", g=num_groups, n=1, i=head_dim)
    return q.T, k.T, v.T


def convert(src, tgt, tp_size):
    """Convert InternLM2 huggingface checkpoints to Llama-style."""
    print("Loading origin checkpoints...")
    hf_states = []
    hf_state_names = []
    remain_files = []
    for filename in tqdm(os.listdir(src)):
        if not filename.endswith(".bin"):
            remain_files.append(filename)
            continue
        hf_state_names.append(filename)
        hf_states.append(torch.load(os.path.join(src, filename)))

    print("Convert InternLM2 huggingface checkpoints to Llama...")

    config = AutoConfig.from_pretrained(src, trust_remote_code=True)

    q_per_kv = config.num_attention_heads // config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    num_heads = config.num_attention_heads
    num_heads_per_tp = num_heads // tp_size
    num_groups = num_heads_per_tp // q_per_kv

    for states in tqdm(hf_states):
        tmp_states = defaultdict(defaultdict)
        for k, v in states.copy().items():
            if "wqkv" in k:
                wqkvs = v.chunk(tp_size, 0)
                for i in range(tp_size):
                    wq, wk, wv = split_wqkv(wqkvs[i], num_groups, q_per_kv, head_dim)

                    _prefix = k.split("attention")[0]
                    wq_key = _prefix + "self_attn.q_proj.weight"
                    wk_key = _prefix + "self_attn.k_proj.weight"
                    wv_key = _prefix + "self_attn.v_proj.weight"

                    tmp_states[wq_key][i] = wq.clone()
                    tmp_states[wk_key][i] = wk.clone()
                    tmp_states[wv_key][i] = wv.clone()

            elif "attention.wo" in k:
                new_k = k.replace("attention.wo", "self_attn.o_proj")
                states[new_k] = v
                del states[k]
            elif "feed_forward.w1" in k:
                new_k = k.replace("feed_forward.w1", "mlp.gate_proj")
                states[new_k] = v
                del states[k]
            elif "feed_forward.w2" in k:
                new_k = k.replace("feed_forward.w2", "mlp.up_proj")
                states[new_k] = v
                del states[k]
            elif "feed_forward.w3" in k:
                new_k = k.replace("feed_forward.w3", "mlp.down_proj")
                states[new_k] = v
                del states[k]
            elif "attention_norm" in k:
                new_k = k.replace("attention_norm", "input_layernorm")
                states[new_k] = v
                del states[k]
            elif "ffn_norm" in k:
                new_k = k.replace("ffn_norm", "post_attention_layernorm")
                states[new_k] = v
                del states[k]
            elif "tok_embeddings" in k:
                states["model.embed_tokens.weight"] = v
                del states[k]
            elif "output" in k:
                states["lm_head.weight"] = v
                del states[k]

        for k, v in tmp_states.items():
            states[k] = torch.cat(list(v.values()), dim=0)

    os.makedirs(tgt, exist_ok=True)
    for i, states in enumerate(hf_states):
        print(f"Saving to {os.path.join(tgt, hf_state_names[i])}...", flush=True)
        torch.save(states, os.path.join(tgt, hf_state_names[i]))
    for filename in remain_files:
        print(f"Copying {filename}...", flush=True)
        os.system(f"cp {os.path.join(src, filename)} {tgt}")
    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")
    parser.add_argument("--tp_size", type=int, help="world_size of tensor parallel")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    convert(args.src, args.tgt, args.tp_size)
