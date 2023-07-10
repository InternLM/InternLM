# InternLM Transformers

[English](./README.md) |
[简体中文](./README-zh-Hans.md) 

This folder contains the `InternLM` model in transformers format.

## Weight Conversion

`convert2hf.py` can convert saved training weights into the transformers format with a single command. Execute the command in the root directory of repository:

```bash
python tools/transformers/convert2hf.py --src_folder origin_ckpt/ --tgt_folder hf_ckpt/ --tokenizer ./tools/V7_sft.model
```

Then, you can load it using the `from_pretrained` interface:

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

`intern_moss_example.py` demonstrates an example of how to use LoRA for fine-tuning on the `fnlp/moss-moon-002-sft` dataset.
