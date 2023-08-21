#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoModel

model = AutoModel.from_pretrained("../hf_ckpt/", trust_remote_code=True).cuda()
print(model)
assert model.config.hidden_size == 2048
assert model.config.num_attention_heads == 16
assert model.config.num_hidden_layers == 16
