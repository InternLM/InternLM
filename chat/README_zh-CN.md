# 对话

[English](./README.md) | 简体中文

本文介绍采用 [Transformers](#import-from-transformers)、[ModelScope](#import-from-modelscope)、[Web demos](#dialogue)
对 InternLM2.5-Chat 进行推理。

你还可以进一步了解 InternLM2.5-Chat 采用的[对话格式](./chat_format_zh-CN.md)，以及如何[用 LMDeploy 进行推理或部署服务](./lmdeploy_zh-CN.md)，或者尝试用 [OpenAOE](./openaoe.md) 与多个模型对话。

## 通过 Transformers 加载

通过以下的代码从 Transformers 加载 InternLM 模型 （可修改模型名称替换不同的模型）

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True).cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好！有什么我可以帮助你的吗？
>>> response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
>>> print(response)
```

### 通过 ModelScope 加载

通过以下的代码从 ModelScope 加载 InternLM2.5-Chat 模型 （可修改模型名称替换不同的模型）

```python
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2_5-7b-chat', revision='v1.0.0')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir,device_map="auto",  trust_remote_code=True,torch_dtype=torch.float16)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

## 通过前端网页对话

可以通过以下代码启动一个前端的界面来与 InternLM2.5 Chat 7B 模型进行交互

```bash
pip install streamlit
pip install transformers>=4.38
streamlit run ./web_demo.py
```
