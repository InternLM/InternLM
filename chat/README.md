# Chat

English | [简体中文](./README_zh-CN.md)

This document briefly shows how to use [Transformers](#import-from-transformers), [ModelScope](#import-from-modelscope), and [Web demos](#dialogue) to conduct inference with InternLM2.5-Chat.

You can also know more about the [chatml format](./chat_format.md) and how to use [LMDeploy for inference and model serving](./lmdeploy.md).

## Import from Transformers

To load the InternLM2.5 7B Chat model using Transformers, use the following code:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True).cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "hello", history=[])
>>> print(response)
Hello! How can I help you today?
>>> response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
>>> print(response)
Sure, here are three tips for effective time management:

1. Prioritize tasks based on importance and urgency: Make a list of all your tasks and categorize them into "important and urgent," "important but not urgent," and "not important but urgent." Focus on completing the tasks in the first category before moving on to the others.
2. Use a calendar or planner: Write down deadlines and appointments in a calendar or planner so you don't forget them. This will also help you schedule your time more effectively and avoid overbooking yourself.
3. Minimize distractions: Try to eliminate any potential distractions when working on important tasks. Turn off notifications on your phone, close unnecessary tabs on your computer, and find a quiet place to work if possible.

Remember, good time management skills take practice and patience. Start with small steps and gradually incorporate these habits into your daily routine.
```

## Import from ModelScope

To load the InternLM2.5 Chat model using ModelScope, use the following code:

```python
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2_5-7b-chat')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir,device_map="auto",  trust_remote_code=True,torch_dtype=torch.float16)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

## Dialogue

You can interact with the InternLM2.5 Chat model through a frontend interface by running the following code:

```bash
pip install streamlit
pip install transformers>=4.38
streamlit run ./chat/web_demo.py
```

The effect is similar to below:

![demo](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)
