# InternLM

<div align="center">

<img src="./assets//logo.svg" width="200"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">书生·浦语 官网</font></b>
    <sup>
      <a href="https://internlm.intern-ai.org.cn/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    <div>&nbsp;</div>
  </div>

[![license](./assets//license.svg)](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE)
[![evaluation](./assets//compass_support.svg)](https://github.com/internLM/OpenCompass/)
<!-- [![Documentation Status](https://readthedocs.org/projects/internlm/badge/?version=latest)](https://internlm.readthedocs.io/zh_CN/latest/?badge=latest) -->

[📘对话教程](./chat) |
[🛠️智能体教程](./agent) |
[📊评测](./evaluation) |
[👀模型库](./model_cards) |
[🤗HuggingFace](https://huggingface.co/spaces/internlm/internlm2-Chat-7B) |
[🆕Update News](#news) |
[🤔提交反馈](https://github.com/InternLM/InternLM/issues/new)

[English](./README.md) |
[简体中文](./README_zh-CN.md) |

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> 和 <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">微信社区</a>
</p>

## 简介

InternLM 是一个开源的轻量级训练框架，旨在支持大模型训练而无需大量的依赖。通过单一的代码库，它支持在拥有数千个 GPU 的大型集群上进行预训练，并在单个 GPU 上进行微调，同时实现了卓越的性能优化。在1024个 GPU 上训练时，InternLM 可以实现近90%的加速效率。

基于InternLM训练框架，我们已经发布了两个开源的预训练模型：InternLM-7B 和 InternLM-20B。

## 更新

[2024.01.17] 我们发布了 InternLM2-7B 和 InternLM2-20B 以及相关的对话模型，InternLM2 在数理、代码、对话、创作等各方面能力都获得了长足进步，综合性能达到开源模型的领先水平。可以点击 [下面的模型库](#model-zoo)进行下载或者[查看模型文档](./model_cards/)来了解更多细节.

[2023.12.13] 我们更新了 InternLM-7B-Chat 和 InternLM-20B-Chat 模型权重。通过改进微调数据和训练策略，新版对话模型生成的回复质量更高、语言风格更加多元。

[2023.09.20] InternLM-20B 已发布，包括基础版和对话版。

## Model Zoo

| Model | Transformers(HF) | ModelScope(HF) | OpenXLab(HF) | Release Date |
|---------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| **InternLM2 Chat 20B**     | [🤗internlm/internlm-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)         | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b/summary)         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b)     | 2024-01-17   |
| **InternLM2 20B** | [🤗internlm/internlm2-20b](https://huggingface.co/internlm/internlm2-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b) | 2024-01-17 |
| **InternLM2 Chat 20B SFT**     | [🤗internlm/internlm-chat-20b-sft](https://huggingface.co/internlm/internlm2-chat-20b-sft)         | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b-sft/summary)         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft)     | 2024-01-17   |
| **InternLM2 Base 20B** | [🤗internlm/internlm2-base-20b](https://huggingface.co/internlm/internlm2-base-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b) | 2024-01-17 |
| **InternLM2 Chat 7B**      | [🤗internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)           | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b)      | 2024-01-17  |
| **InternLM2 7B**           | [🤗internlm/internlm2-7b](https://huggingface.co/internlm/internlm2-7b)                     | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b/summary)                     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b)           |  2024-01-17   |
| **InternLM2 Chat 7B SFT**      | [🤗internlm/internlm2-chat-7b-sft](https://huggingface.co/internlm/internlm2-chat-7b-sft)           | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b-sft/summary)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft)      | 2024-01-17  |
| **InternLM2 Base 7B**           | [🤗internlm/internlm2-base-7b](https://huggingface.co/internlm/internlm2-base-7b)                     | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b/summary)                     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b)           |  2024-01-17   |

## 使用案例

接下来我们展示使用 [Transformers](#import-from-transformers), [ModelScope](#import-from-modelscope), 和 [Web demo](#dialogue) 进行推理.
对话模型采用了 [chatml 格式](./chat/chat_format.md) 来支持通用对话和智能体应用。

### 通过 Transformers 加载

通过以下的代码从 Transformers 加载 InternLM 模型 （可修改模型名称替换不同的模型）

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True).cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好！有什么我可以帮助你的吗？
>>> response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
>>> print(response)
```

### 通过 ModelScope 加载

通过以下的代码从 ModelScope 加载 InternLM 模型 （可修改模型名称替换不同的模型）

```python
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir,device_map="auto",  trust_remote_code=True,torch_dtype=torch.float16)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

### 通过前端网页对话

可以通过以下代码启动一个前端的界面来与 InternLM Chat 7B 模型进行交互

```bash
pip install streamlit==1.24.0
pip install transformers==4.30.2
streamlit run ./chat/web_demo.py
```

效果如下

![效果](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

### 基于InternLM高性能部署

我们使用 [LMDeploy](https://github.com/InternLM/LMDeploy) 完成 InternLM 的一键部署。

```shell
python3 -m pip install lmdeploy
lmdeploy chat turbomind InternLM/internlm-chat-7b --model-name internlm-chat-7b
```

请参考[部署指南](./chat/lmdeploy.md)了解更多使用案例，更多部署教程则可在[这里](https://github.com/InternLM/LMDeploy)找到。

## 微调&训练

请参考[微调教程](./finetune/)尝试续训或微调 InternLM2。

**注意：**本项目中的全量训练功能已经迁移到了[InternEvo](https://github.com/InternLM/InternEvo)以便捷用户的使用。InternEvo 提供了高效的预训练和微调基建用于训练 InternLM 系列模型。

## 贡献

我们感谢所有的贡献者为改进和提升 InternLM 所作出的努力。非常欢迎社区用户能参与进项目中来。请参考贡献指南来了解参与项目贡献的相关指引。

## 致谢

InternLM 代码库是一款由上海人工智能实验室和来自不同高校、企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望这个工具箱和基准测试可以为社区提供灵活高效的代码工具，供用户微调 InternLM 并开发自己的新模型，从而不断为开源社区提供贡献。特别鸣谢[flash-attention](https://github.com/HazyResearch/flash-attention) 与 [ColossalAI](https://github.com/hpcaitech/ColossalAI) 两项开源项目。

## 开源许可证

本仓库的代码依照 Apache-2.0 协议开源。模型权重对学术研究完全开放，也可申请免费的商业使用授权（[申请表](https://wj.qq.com/s2/12725412/f7c1/)）。其他问题与合作请联系 <internlm@pjlab.org.cn>。

## 引用

```
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```
