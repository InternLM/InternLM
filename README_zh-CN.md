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

[📘商业授权](#开源许可证) |
[🤗HuggingFace](https://huggingface.co/internlm) |
[🆕最新消息](#更新) |
[🤔提交反馈](https://github.com/InternLM/InternLM/issues/new)

[English](./README.md) |
[简体中文](./README_zh-CN.md)

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> 和 <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">微信社区</a>
</p>

## 简介

InternLM2 系列模型在本仓库正式发布，具有如下特性：

- 有效支持20万字超长上下文：模型在 20 万字长输入中几乎完美地实现长文“大海捞针”，而且在 LongBench 和 L-Eval 等长文任务中的表现也达到开源模型中的领先水平。 可以通过 [LMDeploy](./chat/lmdeploy_zh_cn.md) 尝试20万字超长上下文推理。
- 综合性能全面提升：各能力维度相比上一代模型全面进步，在推理、数学、代码、对话体验、指令遵循和创意写作等方面的能力提升尤为显著，综合性能达到同量级开源模型的领先水平，在重点能力评测上 InternLM2-Chat-20B 能比肩甚至超越 ChatGPT （GPT-3.5）。
- 代码解释器与数据分析：在配合代码解释器（code-interpreter）的条件下，InternLM2-Chat-20B 在 GSM8K 和 MATH 上可以达到和 GPT-4 相仿的水平。基于在数理和工具方面强大的基础能力，InternLM2-Chat 提供了实用的数据分析能力。
- 工具调用能力整体升级：基于更强和更具有泛化性的指令理解、工具筛选与结果反思等能力，新版模型可以更可靠地支持复杂智能体的搭建，支持对工具进行有效的多轮调用，完成较复杂的任务。可以查看更多[样例](./agent/)。

## 更新

\[2024.01.23\] 我们发布了 InternLM2-Math-7B 和 InternLM2-Math-20B 以及相关的对话模型。InternLM-Math以较小的尺寸超过了ChatGPT的表现。可以点击[InternLM-Math](https://github.com/InternLM/internlm-math)进行下载，并了解详情。

\[2024.01.17\] 我们发布了 InternLM2-7B 和 InternLM2-20B 以及相关的对话模型，InternLM2 在数理、代码、对话、创作等各方面能力都获得了长足进步，综合性能达到开源模型的领先水平。可以点击[下面的模型库](#model-zoo)进行下载或者[查看模型文档](./model_cards/)来了解更多细节.

\[2023.12.13\] 我们更新了 InternLM-7B-Chat 和 InternLM-20B-Chat 模型权重。通过改进微调数据和训练策略，新版对话模型生成的回复质量更高、语言风格更加多元。

\[2023.09.20\] InternLM-20B 已发布，包括基础版和对话版。

## Model Zoo

| Model                      | Transformers(HF)                           | ModelScope(HF)                           | OpenXLab(HF)                           | OpenXLab(Origin)                           | Release Date |
| -------------------------- | ------------------------------------------ | ---------------------------------------- | -------------------------------------- | ------------------------------------------ | ------------ |
| **InternLM2-Base-7B**      | [🤗internlm2-base-7b](https://huggingface.co/internlm/internlm2-base-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b-original) | 2024-01-17   |
| **InternLM2-7B**           | [🤗internlm2-7b](https://huggingface.co/internlm/internlm2-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b-original) | 2024-01-17   |
| **InternLM2-Chat-7B-SFT**  | [🤗internlm2-chat-7b-sft](https://huggingface.co/internlm/internlm2-chat-7b-sft) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b-sft/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft-original) | 2024-01-17   |
| **InternLM2-Chat-7B**      | [🤗internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-original) | 2024-01-17   |
| **InternLM2-Base-20B**     | [🤗internlm2-base-20b](https://huggingface.co/internlm/internlm2-base-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b-original) | 2024-01-17   |
| **InternLM2-20B**          | [🤗internlm2-20b](https://huggingface.co/internlm/internlm2-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b-original) | 2024-01-17   |
| **InternLM2-Chat-20B-SFT** | [🤗internlm2-chat-20b-sft](https://huggingface.co/internlm/internlm2-chat-20b-sft) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b-sft/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft-original) | 2024-01-17   |
| **InternLM2-Chat-20B**     | [🤗internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-original) | 2024-01-17   |

**模型说明：**

在此次发布中，InternLM2 包含两种模型规格：7B 和 20B。7B 为轻量级的研究和应用提供了一个轻便但性能不俗的模型，20B 模型的综合性能更为强劲，可以有效支持更加复杂的实用场景。每个规格不同模型关系如下图所示：

![](https://internlm.oss-cn-shanghai.aliyuncs.com/series.png)

1. **InternLM2-Base**：高质量和具有很强可塑性的模型基座，是模型进行深度领域适配的高质量起点。
2. **InternLM2**：进一步在大规模无标签数据上进行预训练，并结合特定领域的增强语料库进行训练，在评测中成绩优异，同时保持了很好的通用语言能力，是我们推荐的在大部分应用中考虑选用的优秀基座。
3. **InternLM2-Chat-SFT**: 基于 InternLM2-Base 模型进行了有监督微调，是 InternLM2-Chat 模型的中间版本。我们将它们开源以助力社区在对齐方面的研究。
4. **InternLM2-Chat**: 在 InternLM2-Chat-SFT 的基础上进行了 online RLHF 以进一步对齐. InternLM2-Chat 面向对话交互进行了优化，具有较好的指令遵循、共情聊天和调用工具等的能力，是我们推荐直接用于下游应用的模型。

**局限性：** 尽管在训练过程中我们非常注重模型的安全性，尽力促使模型输出符合伦理和法律要求的文本，但受限于模型大小以及概率生成范式，模型可能会产生各种不符合预期的输出，例如回复内容包含偏见、歧视等有害内容，请勿传播这些内容。由于传播不良信息导致的任何后果，本项目不承担责任。

**补充说明：** 上表中的 `HF` 表示对应模型为 HuggingFace 平台提供的 [transformers](https://github.com/huggingface/transformers) 框架格式；`Origin` 则表示对应模型为我们 InternLM 团队的 [InternEvo](https://github.com/InternLM/InternEvo) 框架格式。

## 性能

### 客观评测

| Dataset          | Baichuan2-7B-Chat | Mistral-7B-Instruct-v0.2 | Qwen-7B-Chat | InternLM2-Chat-7B | ChatGLM3-6B | Baichuan2-13B-Chat | Mixtral-8x7B-Instruct-v0.1 | Qwen-14B-Chat | InternLM2-Chat-20B |
| ---------------- | ----------------- | ------------------------ | ------------ | ----------------- | ----------- | ------------------ | -------------------------- | ------------- | ------------------ |
| MMLU             | 50.1              | 59.2                     | 57.1         | 63.7              | 58.0        | 56.6               | 70.3                       | 66.7          | 66.5               |
| CMMLU            | 53.4              | 42.0                     | 57.9         | 63.0              | 57.8        | 54.8               | 50.6                       | 68.1          | 65.1               |
| AGIEval          | 35.3              | 34.5                     | 39.7         | 47.2              | 44.2        | 40.0               | 41.7                       | 46.5          | 50.3               |
| C-Eval           | 53.9              | 42.4                     | 59.8         | 60.8              | 59.1        | 56.3               | 54.0                       | 71.5          | 63.0               |
| TrivialQA        | 37.6              | 35.0                     | 46.1         | 50.8              | 38.1        | 40.3               | 57.7                       | 54.5          | 53.9               |
| NaturalQuestions | 12.8              | 8.1                      | 18.6         | 24.1              | 14.0        | 12.7               | 22.5                       | 22.9          | 25.9               |
| C3               | 78.5              | 66.9                     | 84.4         | 91.5              | 79.3        | 84.4               | 82.1                       | 91.5          | 93.5               |
| CMRC             | 8.1               | 5.6                      | 14.6         | 63.8              | 43.2        | 27.8               | 5.3                        | 13.0          | 50.4               |
| WinoGrande       | 49.9              | 50.8                     | 54.2         | 65.8              | 61.7        | 50.9               | 60.9                       | 55.7          | 74.8               |
| BBH              | 35.9              | 46.5                     | 45.5         | 61.2              | 56.0        | 42.5               | 57.3                       | 55.8          | 68.3               |
| GSM-8K           | 32.4              | 48.3                     | 44.1         | 70.7              | 53.8        | 56.0               | 71.7                       | 57.7          | 79.6               |
| Math             | 5.7               | 8.6                      | 12.0         | 23.0              | 20.4        | 4.3                | 22.5                       | 27.6          | 31.9               |
| HumanEval        | 17.7              | 35.4                     | 36.0         | 59.8              | 52.4        | 19.5               | 37.8                       | 40.9          | 67.1               |
| MBPP             | 37.7              | 25.7                     | 33.9         | 51.4              | 55.6        | 40.9               | 40.9                       | 30.0          | 65.8               |

- MBPP性能使用的是MBPP(Sanitized)版本数据集

### 主观评测

- 我们评测了InternLM2-Chat在[AlpacaEval 2.0](https://tatsu-lab.github.io/alpaca_eval/) 上的性能，结果表明InternLM2-Chat在AlpacaEval上已经超过了 Claude 2, GPT-4(0613) 和  Gemini Pro.

| Model Name         | Win Rate | Length |
| ------------------ | -------- | ------ |
| GPT-4 Turbo        | 50.00%   | 2049   |
| GPT-4              | 23.58%   | 1365   |
| GPT-4 0314         | 22.07%   | 1371   |
| Mistral Medium     | 21.86%   | 1500   |
| XwinLM 70b V0.1    | 21.81%   | 1775   |
| InternLM2 Chat 20B | 21.75%   | 2373   |
| Mixtral 8x7B v0.1  | 18.26%   | 1465   |
| Claude 2           | 17.19%   | 1069   |
| Gemini Pro         | 16.85%   | 1315   |
| GPT-4 0613         | 15.76%   | 1140   |
| Claude 2.1         | 15.73%   | 1096   |

- 性能数据截止2024-01-17

## 依赖

- Python >= 3.8
- PyTorch >= 1.12.0 (推荐 2.0.0 和更高版本)
- Transformers >= 4.34

## 使用案例

接下来我们展示使用 [Transformers](#import-from-transformers)，[ModelScope](#import-from-modelscope) 和 [Web demo](#dialogue) 进行推理。
对话模型采用了 [chatml 格式](./chat/chat_format.md) 来支持通用对话和智能体应用。
为了保障更好的使用效果，在用 [Transformers](#import-from-transformers) 或 [ModelScope](#import-from-modelscope) 进行推理前，请确保安装的 transformers 库版本满足以下要求：

```
transformers >= 4.34
```

### 通过 Transformers 加载

通过以下的代码从 Transformers 加载 InternLM2-7B-Chat 模型 （可修改模型名称替换不同的模型）

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
# 设置`torch_dtype=torch.float16`来将模型精度指定为torch.float16，否则可能会因为您的硬件原因造成显存不足的问题。
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", device_map="auto",trust_remote_code=True, torch_dtype=torch.float16)
# (可选) 如果在低资源设备上，可以通过bitsandbytes加载4-bit或8-bit量化的模型，进一步节省GPU显存.
  # 4-bit 量化的 InternLM 7B 大约会消耗 8GB 显存.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# 模型输出：你好！有什么我可以帮助你的吗？
response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
print(response)
```

### 通过 ModelScope 加载

通过以下的代码从 ModelScope 加载 InternLM 模型 （可修改模型名称替换不同的模型）

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# (可选) 如果在低资源设备上，可以通过bitsandbytes加载4-bit或8-bit量化的模型，进一步节省GPU显存.
  # 4-bit 量化的 InternLM 7B 大约会消耗 8GB 显存.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

### 通过前端网页对话

可以通过以下代码启动一个前端的界面来与 InternLM Chat 7B 模型进行交互

```bash
pip install streamlit
pip install transformers>=4.34
streamlit run ./chat/web_demo.py
```

### 基于 InternLM 高性能部署

我们使用 [LMDeploy](https://github.com/InternLM/LMDeploy) 完成 InternLM 的一键部署。

通过 `pip install lmdeploy>=0.2.1` 安装 LMDeploy 之后，只需 4 行代码，就可以实现离线批处理：

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2-chat-7b")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

请参考[部署指南](./chat/lmdeploy.md)了解更多使用案例，更多部署教程则可在[这里](https://github.com/InternLM/LMDeploy)找到。

## 微调&训练

请参考[微调教程](./finetune/)尝试续训或微调 InternLM2。

**注意：** 本项目中的全量训练功能已经迁移到了 [InternEvo](https://github.com/InternLM/InternEvo) 以便用户使用。InternEvo 提供了高效的预训练和微调基建用于训练 InternLM 系列模型。

## 评测

我们使用 [OpenCompass](https://github.com/open-compass/opencompass) 进行模型评估。在 InternLM-2 中，我们主要标准客观评估、长文评估（大海捞针）、数据污染评估、智能体评估和主观评估。

### 标准客观评测

请按照 [OpenCompass 教程](https://opencompass.readthedocs.io/zh-cn/latest/get_started/installation.html) 进行客观评测。我们通常在 Base 模型上使用 ppl 进行多项选择题评测，在 Chat 模型上使用 gen 进行所有问题的答案生成和评测。

### 长文评估（大海捞针）

有关 `大海捞针` 评估的教程，请参阅 [文档](https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/needleinahaystack_eval.md) 中的教程。

### 数据污染评估

要了解更多关于数据污染评估的信息，请查看 [污染评估](https://opencompass.readthedocs.io/en/latest/advanced_guides/contamination_eval.html)。

### 智能体评估

- 要评估大模型的工具利用能力，请使用 [T-Eval](https://github.com/open-compass/T-Eval) 进行评测。
- 对于代码解释器评估，请使用 [gsm-8k-agent](https://github.com/open-compass/opencompass/blob/main/configs/datasets/gsm8k/gsm8k_agent_gen_be1606.py) 提供的配置进行评估。此外，您还需要安装 [Lagent](https://github.com/InternLM/lagent)。

### 主观评估

- 请按照 [教程](https://opencompass.readthedocs.io/en/latest/advanced_guides/subjective_evaluation.html) 进行主观评估。

## 贡献

我们感谢所有的贡献者为改进和提升 InternLM 所作出的努力。非常欢迎社区用户能参与进项目中来。请参考贡献指南来了解参与项目贡献的相关指引。

## 致谢

InternLM 代码库是一款由上海人工智能实验室和来自不同高校、企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供新功能支持的贡献者，以及提供宝贵反馈意见的用户。我们希望这个工具箱和基准测试可以为社区提供灵活高效的代码工具，供用户微调 InternLM 并开发自己的新模型，从而不断为开源社区提供贡献。特别鸣谢 [flash-attention](https://github.com/HazyResearch/flash-attention) 与 [ColossalAI](https://github.com/hpcaitech/ColossalAI) 两项开源项目。

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
