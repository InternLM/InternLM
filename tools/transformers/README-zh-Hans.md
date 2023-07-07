# InternLM Transformers

[English](./README.md) |
[简体中文](./README-zh-Hans.md) 

该文件夹下包含了 transformers 格式的 `InternLM` 模型。


## 权重转换

`convert2hf.py` 可以将训练保存的权重一键转换为 transformers 格式。

```bash
python convert2hf.py --src_folder origin_ckpt/ --tgt_folder hf_ckpt/ --tokenizer ../v7_sft.model
```

然后可以使用 `from_pretrained` 接口加载：

```python
from modeling_internlm import InternLMForCausalLM

model = InternForCausalLM.from_pretrained("hf_ckpt/")
```


`intern_moss_example.py` 展示了如何使用 LoRA 来在 `fnlp/moss-moon-002-sft` 数据集上进行微调的样例。
