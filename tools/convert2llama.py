# Copyright (c) InternLM. All rights reserved.
import argparse
import json
import os

import torch
from einops import rearrange
from tqdm import tqdm
from transformers import AutoConfig, LlamaConfig, LlamaTokenizer


def weight_load(fp, **kwargs):
    """Load weights from a file."""
    is_safetensors = kwargs.pop('is_safetensors', False)

    if is_safetensors:
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError(
                'Before loading ckpts in the `safetensors` format, '
                'please install the `safetensors` package first.')

        model = safe_open(fp, framework='pt')
        state_dict = {}
        for k in model.keys():
            state_dict[k] = model.get_tensor(k)
        return state_dict

    else:
        return torch.load(fp, **kwargs)


def save_conifg(config, tgt):
    config_dict = config.to_dict()
    unnecessary_keys = [
        '_name_or_path',
        'auto_map',
        'transformers_version',
        'model_type',
        'architectures',
        'tokenizer_class',
        'attn_implementation',
    ]
    for k in unnecessary_keys:
        config_dict.pop(k, None)
    config_dict['attention_bias'] = config_dict.pop('bias')
    config_dict['architectures'] = ['LlamaForCausalLM']
    llama_config = LlamaConfig(**config_dict)
    llama_config.save_pretrained(tgt)


def convert(src, tgt):
    """Convert InternLM2 huggingface checkpoints to Llama-style."""

    print('Convert InternLM2 huggingface checkpoints to Llama...')

    config = AutoConfig.from_pretrained(src, trust_remote_code=True)
    assert not config.bias, 'Cannot convert InternLM Model with bias to LLaMA.'

    head_dim = config.hidden_size // config.num_attention_heads
    num_key_value_groups = \
        config.num_attention_heads // config.num_key_value_heads

    # load index json file
    index_file = 'pytorch_model.bin.index.json'
    if os.path.exists(os.path.join(src, index_file)):
        with open(os.path.join(src, index_file)) as fp:
            index_dict = json.load(fp)
            index_dict['weight_map'] = {}
    else:
        index_file = 'model.safetensors.index.json'
        if os.path.exists(os.path.join(src, index_file)):
            with open(os.path.join(src, index_file)) as fp:
                index_dict = json.load(fp)
                index_dict['weight_map'] = {}
        else:
            index_dict = None

    os.makedirs(tgt, exist_ok=True)
    for filename in tqdm(os.listdir(src)):
        if not any(filename.endswith(ext) for ext in ('.bin', '.safetensors')):
            continue

        print(f'Loading {os.path.join(src, filename)}...', flush=True)
        states = weight_load(os.path.join(src, filename),
                             is_safetensors=filename.endswith('.safetensors'))

        llama_states = {}
        for k, v in states.copy().items():
            if 'wqkv' in k:
                v = rearrange(
                    v,
                    '(h gs d) dim -> h gs d dim',
                    gs=2 + num_key_value_groups,
                    d=head_dim,
                )
                wq, wk, wv = torch.split(v, [num_key_value_groups, 1, 1],
                                         dim=1)
                wq = rearrange(wq, 'h gs d dim -> (h gs d) dim')
                wk = rearrange(wk, 'h gs d dim -> (h gs d) dim')
                wv = rearrange(wv, 'h gs d dim -> (h gs d) dim')
                _prefix = k.split('attention')[0]
                wq_key = _prefix + 'self_attn.q_proj.weight'
                wk_key = _prefix + 'self_attn.k_proj.weight'
                wv_key = _prefix + 'self_attn.v_proj.weight'
                llama_states[wq_key] = wq.clone()
                llama_states[wk_key] = wk.clone()
                llama_states[wv_key] = wv.clone()

            elif 'attention.wo' in k:
                new_k = k.replace('attention.wo', 'self_attn.o_proj')
                llama_states[new_k] = v
            elif 'feed_forward.w1' in k:
                new_k = k.replace('feed_forward.w1', 'mlp.gate_proj')
                llama_states[new_k] = v
            elif 'feed_forward.w2' in k:
                new_k = k.replace('feed_forward.w2', 'mlp.down_proj')
                llama_states[new_k] = v
            elif 'feed_forward.w3' in k:
                new_k = k.replace('feed_forward.w3', 'mlp.up_proj')
                llama_states[new_k] = v
            elif 'attention_norm' in k:
                new_k = k.replace('attention_norm', 'input_layernorm')
                llama_states[new_k] = v
            elif 'ffn_norm' in k:
                new_k = k.replace('ffn_norm', 'post_attention_layernorm')
                llama_states[new_k] = v
            elif 'tok_embeddings' in k:
                llama_states['model.embed_tokens.weight'] = v
            elif 'output' in k:
                llama_states['lm_head.weight'] = v
            else:
                llama_states[k] = v

        if index_dict is not None:
            for k in llama_states:
                index_dict['weight_map'][k] = filename

        print(f'Saving to {os.path.join(tgt, filename)}...', flush=True)
        if filename.endswith('.safetensors'):
            from safetensors.torch import save_file
            save_file(llama_states,
                      os.path.join(tgt, filename),
                      metadata={'format': 'pt'})
        else:
            torch.save(llama_states, os.path.join(tgt, filename))
        del states

    print('Saving config and tokenizer...', flush=True)
    # index.json
    if index_dict is not None:
        with open(os.path.join(tgt, index_file), 'w') as fp:
            json.dump(index_dict, fp, indent=2)
    # tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(src)
    tokenizer.init_kwargs.pop('auto_map', None)
    tokenizer.save_pretrained(tgt)
    # config
    save_conifg(config, tgt)

    print('Done!', flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='Input folder')
    parser.add_argument('--tgt', type=str, help='Output folder')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    convert(args.src, args.tgt)
