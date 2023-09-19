import argparse
import math
import json
import os
import re
import tempfile

import torch
from modeling_internlm import InternLMConfig, InternLMForCausalLM
from tokenization_internlm import InternLMTokenizer

NUM_SHARDS = {
    "7B": 1,
}


def convert2hf(model_config, states_tp_pps):

    with tempfile.TemporaryDirectory() as folder:
        states = merge_pp(states_tp_pps)[0]

        if "embedding.word_embeddings.weight" in states:
            embedding_key = "embedding.word_embeddings.weight"
        elif "embedding.weight" in states:
            embedding_key = "embedding.weight"
        else:
            print("Check embedding states'names in below:", flush=True)
            print(list(states.keys()), flush=True)

        dims_per_head = model_config["hidden_size"] // model_config["num_attention_heads"]
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

        current_states = {}

        current_states["model.embed_tokens.weight"] = states.pop(embedding_key)
        current_states["model.norm.weight"] = states.pop("norm.weight")
        current_states["lm_head.weight"] = states.pop("head.weight")

        for i in range(model_config["num_layers"]):
            states.pop(f"blocks.{i}.mixer.rotary_emb.inv_freq", None)

            wqkv = states.pop(f"blocks.{i}.mixer.Wqkv.weight").reshape(
                3, model_config["num_attention_heads"], -1, model_config["hidden_size"]
            )
            bqkv = states.pop(f"blocks.{i}.mixer.Wqkv.bias").reshape(3, model_config["num_attention_heads"], -1)

            current_states[f"model.layers.{i}.self_attn.q_proj.weight"] = wqkv[0].reshape(
                -1, model_config["hidden_size"]
            )
            current_states[f"model.layers.{i}.self_attn.q_proj.bias"] = bqkv[0].reshape(-1)
            current_states[f"model.layers.{i}.self_attn.k_proj.weight"] = wqkv[1].reshape(
                -1, model_config["hidden_size"]
            )
            current_states[f"model.layers.{i}.self_attn.k_proj.bias"] = bqkv[1].reshape(-1)
            current_states[f"model.layers.{i}.self_attn.v_proj.weight"] = wqkv[2].reshape(
                -1, model_config["hidden_size"]
            )
            current_states[f"model.layers.{i}.self_attn.v_proj.bias"] = bqkv[2].reshape(-1)

            current_states[f"model.layers.{i}.self_attn.o_proj.weight"] = states.pop(
                f"blocks.{i}.mixer.out_proj.weight"
            )
            current_states[f"model.layers.{i}.self_attn.o_proj.bias"] = states.pop(f"blocks.{i}.mixer.out_proj.bias")

            current_states[f"model.layers.{i}.mlp.gate_proj.weight"] = states.pop(f"blocks.{i}.mlp.w1.weight")
            current_states[f"model.layers.{i}.mlp.down_proj.weight"] = states.pop(f"blocks.{i}.mlp.w3.weight")
            current_states[f"model.layers.{i}.mlp.up_proj.weight"] = states.pop(f"blocks.{i}.mlp.w2.weight")

            current_states[f"model.layers.{i}.input_layernorm.weight"] = states.pop(f"blocks.{i}.norm1.weight")
            current_states[f"model.layers.{i}.post_attention_layernorm.weight"] = states.pop(f"blocks.{i}.norm2.weight")
            current_states[f"model.layers.{i}.self_attn.rotary_emb.inv_freq"] = inv_freq

        config = InternLMConfig(
            hidden_size=model_config["hidden_size"],
            intermediate_size=compute_intermediate_size(model_config["hidden_size"]),
            num_attention_heads=model_config["num_attention_heads"],
            num_hidden_layers=model_config["num_layers"],
            rms_norm_eps=1e-06,
            bias=True,
        )

        if model_config["vocab_size"] != -1:
            config.vocab_size = model_config["vocab_size"]

        config.save_pretrained(folder)
        torch.save(current_states, os.path.join(folder, "pytorch_model.bin"))

        model = InternLMForCausalLM.from_pretrained(folder, torch_dtype=torch.float16)
        del model.config._name_or_path

    return config, model


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def merge_pp(states_tp_pp):
    max_tp = len(states_tp_pp)
    max_pp = len(states_tp_pp[0])

    full_states = []
    for tp in range(max_tp):
        layer_shift = 0

        tp_states = {}
        for pp in range(max_pp):
            _layer_shift = 0
            states = states_tp_pp[tp][pp]
            keys = list(states.keys())
            for key in keys:
                match = re.search("\.\d+\.", key)
                if match is not None:
                    s, e = match.span()
                    layer_idx = int(key[s + 1 : e - 1]) + layer_shift
                    _layer_shift = max(_layer_shift, int(key[s + 1 : e - 1]))
                    name = key[:s] + f".{layer_idx}." + key[e:]
                    tp_states[name] = states[key]
                else:
                    tp_states[key] = states[key]
            layer_shift += _layer_shift + 1
        full_states.append({(key[6:] if key.startswith("model.") else key): value for key, value in tp_states.items()})
    return full_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', type=str, default='~/test/') # 需要转换为hf格式的checkpoint文件夹
    parser.add_argument('--tgt_folder', type=str, default='~/output/') # 存放转换后checkpoint的目标文件夹
    parser.add_argument('--tokenizer', type=str, default='~/test/tokenizer.model') # Tokenizer 文件的路径
    args = parser.parse_args()

    def load(fp):
        with open(fp, "rb") as f:
            pt_data = torch.load(f, map_location="cpu")
        return pt_data

    folder = args.src_folder
    target_folder = args.tgt_folder
    model_config = load(os.path.join(folder, "model_config.pt"))

    fns = list(os.listdir(folder))

    model_fns = []
    for fn in fns:
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(fn)

    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split("_")
        max_pp = max(max_pp, int(pp[2:]) + 1)
        max_tp = max(max_tp, int(tp[2:]) + 1)

    states_tp_pps = [[]]

    for pp in range(max_pp):
        model_name = f"model_tp0_pp{pp}.pt"
        states = load(os.path.join(folder, model_name))
        states_tp_pps[0].append(states)

    config, model = convert2hf(model_config, states_tp_pps)

    os.makedirs(target_folder, exist_ok=True)
    model.save_pretrained(target_folder, max_shard_size="20GB")
    # TODO There should be a better way to add this.
    with open(os.path.join(target_folder, "config.json")) as fp:
        config_dict = json.load(fp)
    config_dict["auto_map"]["AutoModel"] = "modeling_internlm.InternLMForCausalLM"
    with open(os.path.join(target_folder, "config.json"), "w") as fp:
        json.dump(config_dict, fp, indent=2)

    tokenizer = InternLMTokenizer(args.tokenizer)
    tokenizer.save_pretrained(target_folder)
