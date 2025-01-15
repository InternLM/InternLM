# Chat

English | [简体中文](./README_zh-CN.md)

This document briefly shows how to use [Transformers](#import-from-transformers), [ModelScope](#import-from-modelscope), and [Web demos](#dialogue) to conduct inference with InternLM3-Instruct.

You can also know more about the [chatml format](./chat_format.md) and how to use [LMDeploy for inference and model serving](./lmdeploy.md).

## Import from Transformers

To load the InternLM3-8B-Instruct model using Transformers, use the following code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm3-8b-instruct", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm3-8b-instruct", trust_remote_code=True, torch_dtype=torch.float16)
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM3 8B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained("internlm/internlm3-8b-instruct", device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained("internlm/internlm3-8b-instruct", device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()

messages = [
    {"role": "system", "content": "You are an AI assistant whose name is InternLM."},
    {"role": "user", "content": "Please tell me five scenic spots in Shanghai"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

generated_ids = model.generate(tokenized_chat, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
]
response = tokenizer.batch_decode(generated_ids)[0]
```

## Import from ModelScope

To load the InternLM3-8B-Instruct model using ModelScope, use the following code:

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm3-8b-instruct')
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16)
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM3 8B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()

messages = [
    {"role": "system", "content": "You are an AI assistant whose name is InternLM."},
    {"role": "user", "content": "Please tell me five scenic spots in Shanghai"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

generated_ids = model.generate(tokenized_chat, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
]
response = tokenizer.batch_decode(generated_ids)[0]
```

## Dialogue

You can interact with the InternLM3-8B-Instruct model through a frontend interface by running the following code:

```bash
pip install streamlit
pip install transformers>=4.48
streamlit run ./chat/web_demo.py
```

It supports switching between different inference modes and comparing their responses.

![demo](https://github.com/user-attachments/assets/4953befa-343f-499d-b289-048d982439f3)
