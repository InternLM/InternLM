# 对话

[English](./README.md) | 简体中文

本文介绍采用 [Transformers](#import-from-transformers)、[ModelScope](#import-from-modelscope)、[Web demos](#dialogue)
对 InternLM3-Instruct 进行推理。

你还可以进一步了解 InternLM3-Instruct 采用的[对话格式](./chat_format_zh-CN.md)，以及如何[用 LMDeploy 进行推理或部署服务](./lmdeploy_zh-CN.md)，或者尝试用 [OpenAOE](./openaoe.md) 与多个模型对话。

## 通过 Transformers 加载

通过以下的代码从 Transformers 加载 InternLM 模型 （可修改模型名称替换不同的模型）

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm3-8b-instruct')
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
# 设置`torch_dtype=torch.float16`来将模型精度指定为torch.float16，否则可能会因为您的硬件原因造成显存不足的问题。
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# (可选) 如果在低资源设备上，可以通过bitsandbytes加载4-bit或8-bit量化的模型，进一步节省GPU显存.
  # 4-bit 量化的 InternLM3 8B 大约会消耗 8GB 显存.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
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

### 通过 ModelScope 加载

通过以下的代码从 ModelScope 加载 InternLM3-Instruct 模型 （可修改模型名称替换不同的模型）

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm3-8b-instruct')
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
# 设置`torch_dtype=torch.float16`来将模型精度指定为torch.float16，否则可能会因为您的硬件原因造成显存不足的问题。
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# (可选) 如果在低资源设备上，可以通过bitsandbytes加载4-bit或8-bit量化的模型，进一步节省GPU显存.
  # 4-bit 量化的 InternLM3 8B 大约会消耗 8GB 显存.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
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

## 通过前端网页对话

可以通过以下代码启动一个前端的界面来与 InternLM3-8B-Instruct 模型进行交互

```bash
pip install streamlit
pip install transformers>=4.48
streamlit run ./web_demo.py
```

支持切换不同推理模式，并比较它们的回复

![demo](https://github.com/user-attachments/assets/952e250d-22a6-4544-b8e3-9c21c746d3c7)
