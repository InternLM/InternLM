import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoConfig


def revert(src, tgt, tp_size, embed_split_hidden, adapt_hf, use_flash):
    hf_state = {}
    print("Loading HF checkpoints...")
    for filename in tqdm(os.listdir(src)):
        if not filename.endswith(".bin"):
            continue
        hf_state.update(torch.load(os.path.join(src, filename)))

    print("Reverting HF checkpoints to InternLM...")
    config = AutoConfig.from_pretrained(src, trust_remote_code=True)

    n_heads = config.num_attention_heads
    try:
        n_kv_heads = config.num_key_value_heads
    except AttributeError:
        n_kv_heads = n_heads
    dim = config.hidden_size

    # n_heads_per_shard = n_heads // tp_size
    # dims_per_head = dim // n_heads

    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        if adapt_hf:
            return w
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # revert
    states = [{} for _ in range(tp_size)]
    moe_states = [
        [[{} for _ in range(tp_size)] for _ in range(config.num_experts)] for _ in range(config.num_hidden_layers)
    ]

    # layers
    for layer_i in tqdm(range(config.num_hidden_layers)):
        # no-moe
        for i in range(tp_size):
            states[i][f"model.layers.{layer_i}.attention_norm.weight"] = hf_state[
                f"model.layers.{layer_i}.input_layernorm.weight"
            ].clone()
            states[i][f"model.layers.{layer_i}.ffn_norm.weight"] = hf_state[
                f"model.layers.{layer_i}.post_attention_layernorm.weight"
            ].clone()
            states[i][f"model.layers.{layer_i}.feed_forward.moe_layer.gate.wg.weight"] = hf_state[
                f"model.layers.{layer_i}.mlp.gate.weight"
            ].clone()

        # mha
        wqs = (
            permute(hf_state[f"model.layers.{layer_i}.self_attn.q_proj.weight"])
            # .view(-1, dims_per_head, dim)
            .chunk(tp_size, 0)
        )
        wks = (
            permute(hf_state[f"model.layers.{layer_i}.self_attn.k_proj.weight"], n_kv_heads, -1, dim)
            # .view(-1, dims_per_head, dim)
            .chunk(tp_size, 0)
        )
        wvs = (
            hf_state[f"model.layers.{layer_i}.self_attn.v_proj.weight"]
            # .view(-1, dims_per_head, dim)
            .chunk(tp_size, 0)
        )
        wos = hf_state[f"model.layers.{layer_i}.self_attn.o_proj.weight"].chunk(tp_size, 1)
        for i in range(tp_size):
            states[i][f"model.layers.{layer_i}.attention.wq.weight"] = wqs[i].reshape(-1, dim).clone()
            states[i][f"model.layers.{layer_i}.attention.wk.weight"] = wks[i].reshape(-1, dim).clone()
            states[i][f"model.layers.{layer_i}.attention.wv.weight"] = wvs[i].reshape(-1, dim).clone()
            states[i][f"model.layers.{layer_i}.attention.wo.weight"] = wos[i].clone()

        # moe
        for expert_id in range(config.num_experts):
            w1s = hf_state[f"model.layers.{layer_i}.mlp.experts.{expert_id}.w1.weight"].chunk(tp_size, 0)
            w2s = hf_state[f"model.layers.{layer_i}.mlp.experts.{expert_id}.w3.weight"].chunk(tp_size, 0)
            w3s = hf_state[f"model.layers.{layer_i}.mlp.experts.{expert_id}.w2.weight"].chunk(tp_size, 1)
            for i in range(tp_size):
                moe_states[layer_i][expert_id][i][
                    f"model.layers.{layer_i}.feed_forward.moe_layer.experts.experts.{expert_id}.w1.weight"
                ] = w1s[i].clone()
                moe_states[layer_i][expert_id][i][
                    f"model.layers.{layer_i}.feed_forward.moe_layer.experts.experts.{expert_id}.w2.weight"
                ] = w2s[i].clone()
                moe_states[layer_i][expert_id][i][
                    f"model.layers.{layer_i}.feed_forward.moe_layer.experts.experts.{expert_id}.w3.weight"
                ] = w3s[i].clone()

    if embed_split_hidden:
        embeds = hf_state["model.embed_tokens.weight"].chunk(tp_size, 1)
        states[i]["model.tok_embeddings.weight"] = embeds[i].clone()
    else:
        embeds = hf_state["model.embed_tokens.weight"].chunk(tp_size, 0)
        states[i]["model.tok_embeddings.word_embeddings.weight"] = embeds[i].clone()

    outputs = hf_state["lm_head.weight"].chunk(tp_size, 0)
    for i in range(tp_size):
        states[i]["model.norm.weight"] = hf_state["model.norm.weight"].clone()
        states[i]["model.output.weight"] = outputs[i].clone()

    mlp_ratio = round((config.intermediate_size - 255) / config.hidden_size + 0.01, 2)
    if "rotary" in config.to_dict():
        rope_base = config.rotary["base"]
    elif "rope_theta" in config.to_dict():
        rope_base = config.rope_theta
    else:
        rope_base = 10000
    model_config = dict(
        num_attention_heads=n_heads,
        embed_split_hidden=embed_split_hidden,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        norm_type="rmsnorm",
        layer_norm_epsilon=config.rms_norm_eps,
        no_bias=True,
        mlp_ratio=mlp_ratio,
        num_kv_attention_heads=n_kv_heads,
        dtype=config.torch_dtype,
        # norm_head=False,
        adapt_hf=adapt_hf,
        use_flash_attn=use_flash,
        rope_base=rope_base,
        num_experts=config.num_experts,
        moe_gate_k=config.num_experts_per_token,
    )
    print("Model Config:", model_config)

    # save
    os.makedirs(tgt, exist_ok=True)
    print(f"Saving to {tgt}...")
    for tp in tqdm(range(tp_size)):
        torch.save(states[tp], os.path.join(tgt, f"model_tp{tp}_pp0.pt"))
    for moe_layer_id in range(config.num_hidden_layers):
        for expert_id in range(config.num_experts):
            for tp in tqdm(range(tp_size)):
                torch.save(
                    moe_states[moe_layer_id][expert_id][tp],
                    os.path.join(tgt, f"model_moe_layer{moe_layer_id}_expert{expert_id}_tp{tp}.pt"),
                )
    torch.save(model_config, os.path.join(tgt, "model_config.pt"))


def print_args(args):
    print("-------------- Arguments --------------")
    print(f"Source Path: {args.src}")
    print(f"Target Path: {args.tgt}")
    print(f"TP Size: {args.tp_size}")
    print(f"Embeb Split Hidden: {args.embed_split}")
    print(f"Adapt HF: {args.adapt_hf}")
    print(f"Use Flash Attn: {args.use_flash}")
    print("---------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", type=str, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")
    parser.add_argument("--tp_size", type=int, help="world_size of tensor parallel")
    parser.add_argument("--embed_split", action="store_true", help="embed_split_hidden of InternLM")
    parser.add_argument("--adapt_hf", action="store_true", help="adapt_hf of InternLM")
    parser.add_argument("--use_flash", action="store_true", help="use_flash_attn of InternLM")

    args = parser.parse_args()

    return args


# download ckpt from https://huggingface.co/DiscoResearch/mixtral-7b-8expert and
# srun -p llm_s python tools/transformers/mixtral2llamamoe.py --src ./ckpt/mixtral-7b-8expert/ --tgt ckpt --tp_size {tp}
if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    revert(args.src, args.tgt, args.tp_size, args.embed_split, args.adapt_hf, args.use_flash)
