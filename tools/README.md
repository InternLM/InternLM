# InternLM2 tools

## 1. Convert to LLaMA

We offer the `convert2llama.py`, designed to seamlessly transform InternLM2 (HF format) into LLaMA (HF format). Here, HF refers to the format used by HuggingFace Transformers.

### Usage

```
python convert2llama.py --src /path/to/internlm2/ckpt --tgt /path/to/target/ckpt
```

### Note

While the `convert2llama.py` tool is available, we still advise opting for InternLM2 when practical, chiefly due to its superior efficiency. InternLM2, which is adapted from LLaMA, streamlines the process by integrating the `Wq`, `Wk`, `Wv` weight matrices into a single matrix `Wqkv`. This integration leads to approximately a **5%** speed increase during training. Given the substantial costs associated with pre-training, this efficiency boost can result in significant savings.
