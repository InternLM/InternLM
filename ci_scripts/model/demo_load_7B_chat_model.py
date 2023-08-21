#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
assert len(response) != 0
response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
print(response)
assert len(response) != 0
