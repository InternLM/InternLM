# InternLM

<div align="center">

<img src="./doc/imgs/logo.svg" width="200"/>
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

[![license](./doc/imgs/license.svg)](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE)
[![evaluation](./doc/imgs/compass_support.svg)](https://github.com/internLM/OpenCompass/)
[![Documentation Status](https://readthedocs.org/projects/internlm/badge/?version=latest)](https://internlm.readthedocs.io/zh_CN/latest/?badge=latest)

[📘使用文档](./doc/usage.md) |
[🛠️安装教程](./doc/install.md) |
[📊训练性能](./doc/train_performance.md) |
[👀模型库](#model-zoo) |
[🤗HuggingFace](https://huggingface.co/spaces/internlm/InternLM-Chat-7B) |
[🆕Update News](./CHANGE_LOG.md) |
[🤔Reporting Issues](https://github.com/InternLM/InternLM/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> 和 <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">微信社区</a>
</p>

## 简介

InternLM 是一个开源的轻量级训练框架，旨在支持大模型训练而无需大量的依赖。通过单一的代码库，它支持在拥有数千个 GPU 的大型集群上进行预训练，并在单个 GPU 上进行微调，同时实现了卓越的性能优化。在1024个 GPU 上训练时，InternLM 可以实现近90%的加速效率。

基于InternLM训练框架，我们已经发布了两个开源的预训练模型：InternLM-7B 和 InternLM-20B。

## 更新

[20230920] InternLM-20B 已发布，包括基础版和对话版。  
[20230822] InternLM-7B-Chat v1.1 已发布，增加了代码解释器和函数调用能力。您可以使用 [Lagent](https://github.com/InternLM/lagent) 进行尝试。


## Model Zoo

我们的模型在三个平台上发布：Transformers、ModelScope 和 OpenXLab。

| Model                     | Transformers                        | ModelScope                                                                                                                        | OpenXLab                                                                              |发布日期 |
|---------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| **InternLM Chat 20B**     | [🤗internlm/internlm-chat-20b](https://huggingface.co/internlm/internlm-20b-chat)         | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b-chat/summary)         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-20b)     | 2023-09-20   |
| **InternLM 20B**          | [🤗internlm/internlm-20b](https://huggingface.co/internlm/internlm-20b)                   | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary)                   | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-20b)          | 2023-09-20   |
| **InternLM Chat 7B v1.1** | [🤗internlm/internlm-chat-7b-v1.1](https://huggingface.co/internlm/internlm-chat-7b-v1.1) | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-7b-v1_1](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-v1_1/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-v1.1) | 2023-08-22   |
| **InternLM 7B**           | [🤗internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b)                     | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary)                     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-7b)           | 2023-07-06   |
| **InternLM Chat 7B**      | [🤗internlm/internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)           | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b/summary)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b)      | 2023-07-06   |
| **InternLM Chat 7B 8k**   | [🤗internlm/internlm-chat-7b-8k](https://huggingface.co/internlm/internlm-chat-7b-8k)     | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-7b-8k](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-8k/summary)     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-8k)   | 2023-07-06   |


<details> 
<summary> InternLM-20B </summary>

#### 简介
InternLM-20B 在超过 **2.3T** Tokens 包含高质量英文、中文和代码的数据上进行预训练，其中 Chat 版本还经过了 SFT 和 RLHF 训练，使其能够更好、更安全地满足用户的需求。  

InternLM 20B 在模型结构上选择了深结构，InternLM-20B 的层数设定为60层，超过常规7B和13B模型所使用的32层或者40层。在参数受限的情况下，提高层数有利于提高模型的综合能力。此外，相较于InternLM-7B，InternLM-20B使用的预训练数据经过了更高质量的清洗，并补充了高知识密度和用于强化理解和推理能力的训练数据。因此，它在理解能力、推理能力、数学能力、编程能力等考验语言模型技术水平的方面都得到了显著提升。总体而言，InternLM-20B具有以下的特点： 
- 优异的综合性能
- 很强的工具调用功能
- 支持16k语境长度（通过推理时外推）
- 更好的价值对齐

#### 性能对比

在OpenCompass提出的5个能力维度上，InternLM-20B都取得很好的效果（粗体为13B-33B这个量级范围内，各项最佳成绩）

| 能力维度 | Llama-13B | Llama2-13B | Baichuan2-13B | InternLM-20B | Llama-33B | Llama-65B | Llama2-70B |
|----------|-----------|------------|---------------|--------------|-----------|-----------|------------|
| 语言     | 42.5      | 47         | 47.5          | **55**           | 44.6      | 47.1      | 51.6       |
| 知识     | 58.2      | 58.3       | 48.9          | 60.1         | **64**        | 66        | 67.7       |
| 理解     | 45.5      | 50.9       | 58.1          | **67.3**         | 50.6      | 54.2      | 60.8       |
| 推理     | 42.7      | 43.6       | 44.2          | **54.9**         | 46.4      | 49.8      | 55         |
| 学科     | 37.3      | 45.2       | 51.8          | **62.5**         | 47.4      | 49.7      | 57.3       |
| 总平均   | 43.8      | 47.3       | 49.4          | **59.2**         | 48.9      | 51.9      | 57.4       |

下表在一些有重要影响力的典型数据集上比较了主流开源模型的表现

|      | 评测集           | Llama-13B | Llama2-13B | Baichuan2-13B | InternLM-20B | Llama-33B | Llama-65B | Llama2-70B |
|------|------------------|-----------|------------|---------------|--------------|-----------|-----------|------------|
| 学科 | MMLU             | 47.73     | 54.99      | 59.55         | **62.05**        | 58.73     | 63.71     | 69.75      |
|      | C-Eval (val)     | 31.83     | 41.4       | **59.01**         | 58.8         | 37.47     | 40.36     | 50.13      |
|      | AGI-Eval         | 22.03     | 30.93      | 37.37         | **44.58**        | 33.53     | 33.92     | 40.02      |
| 知识 | BoolQ            | 78.75     | 82.42      | 67            | **87.46**        | 84.43     | 86.61     | 87.74      |
|      | TriviaQA         | 52.47     | 59.36      | 46.61         | 57.26        | **66.24**     | 69.79     | 70.71      |
|      | NaturalQuestions | 20.17     | 24.85      | 16.32         | 25.15        | **30.89**     | 33.41     | 34.16      |
| 理解 | CMRC             | 9.26      | 31.59      | 29.85         | **68.78**        | 14.17     | 34.73     | 43.74      |
|      | CSL              | 55        | 58.75      | 63.12         | **65.62**        | 57.5      | 59.38     | 60         |
|      | RACE (middle)    | 53.41     | 63.02      | 68.94         | **86.35**        | 64.55     | 72.35     | 81.55      |
|      | RACE (high)      | 47.63     | 58.86      | 67.18         | **83.28**        | 62.61     | 68.01     | 79.93      |
|      | XSum             | 20.37     | 23.37      | 25.23         | **35.54**        | 20.55     | 19.91     | 25.38      |
| 推理 | WinoGrande       | 64.64     | 64.01      | 67.32         | **69.38**        | 66.85     | 69.38     | 69.77      |
|      | BBH              | 37.93     | 45.62      | 48.98         | **52.51**        | 49.98     | 58.38     | 64.91      |
|      | GSM8K            | 20.32     | 29.57      | **52.62**         | **52.62**        | 42.3      | 54.44     | 63.31      |
|      | PIQA             | 79.71     | 79.76      | 78.07         | 80.25        | **81.34**     | 82.15     | 82.54      |
| 编程 | HumanEval        | 14.02     | 18.9       | 17.07         | **25.61**        | 17.68     | 18.9      | 26.22      |
|      | MBPP             | 20.6      | 26.8       | 30.8          | **35.6**         | 28.4      | 33.6      | 39.6       |

总体而言，InternLM-20B 在综合能力上全面领先于13B量级的开源模型，同时在推理评测集上接近甚至超越Llama-65B的性能。

- 评估结果来自 [OpenCompass 20230920](https://github.com/internLM/OpenCompass/)。
- 由于 [OpenCompass](https://github.com/internLM/OpenCompass/) 的版本迭代，评估数据可能存在数值上的差异，所以请参考 [OpenCompass](https://github.com/internLM/OpenCompass/) 的最新评估结果。

</details>


<details> 
<summary> InternLM-7B </summary>

#### 模型更新
[20230822] 通过使用更丰富的SFT类型数据，InternLM-7B-Chat v1.1模型支持代码解释和函数调用。模型结构与代码没有任何变化，因此可以使用与InternLM-7B-Chat完全一样的方式使用更强大的InternLM-7B-Chat v1.1。

#### 简介
InternLM-7B 包含了一个拥有70亿参数的基础模型和一个为实际场景量身定制的对话模型。该模型具有以下特点：

- 它利用数万亿的高质量令牌进行训练，建立了一个强大的知识库。
- 它支持8k的上下文窗口长度，使得输入序列更长并增强了推理能力。
- 它为用户提供了一个多功能的工具集，使用户能够灵活地构建自己的工作流程。

#### 性能对比

我们使用开源评测工具 [OpenCompass](https://github.com/internLM/OpenCompass/) 从学科综合能力、语言能力、知识能力、推理能力、理解能力五大能力维度对InternLM开展全面评测，部分评测结果如下表所示，欢迎访问[OpenCompass 榜单](https://opencompass.org.cn/rank)获取更多的评测结果。

| 数据集\模型           |  **InternLM-Chat-7B** |  **InternLM-7B**  |  LLaMA-7B | Baichuan-7B | ChatGLM2-6B | Alpaca-7B | Vicuna-7B |
| -------------------- | --------------------- | ---------------- | --------- |  --------- | ------------ | --------- | ---------- |
| C-Eval(Val)          |      53.2             |        53.4       | 24.2      | 42.7       |  50.9       |  28.9     | 31.2     |
| MMLU                 |      50.8             |       51.0        | 35.2*     |  41.5      |  46.0       |  39.7     | 47.3     |
| AGIEval              |      42.5             |       37.6        | 20.8      | 24.6       |  39.0       | 24.1      | 26.4     |
| CommonSenseQA        |      75.2             |      59.5         | 65.0      | 58.8       | 60.0        | 68.7      | 66.7     |
| BUSTM                |      74.3             |       50.6        | 48.5      | 51.3        | 55.0        | 48.8      | 62.5     |
| CLUEWSC              |      78.6             |      59.1         |  50.3     |  52.8     |  59.8     |   50.3    |  52.2     |
| MATH                 |      6.4            |         7.1        |  2.8       | 3.0       | 6.6       |  2.2      | 2.8       |
| GSM8K                |      34.5           |        31.2        | 10.1       | 9.7       | 29.2      |  6.0      | 15.3  |
|  HumanEval           |      14.0           |        10.4        |   14.0     | 9.2       | 9.2       | 9.2       | 11.0  |
| RACE(High)           |      76.3           |        57.4        | 46.9*      | 28.1      | 66.3      | 40.7      | 54.0  |

- 以上评测结果基于 [OpenCompass 20230706](https://github.com/internLM/OpenCompass/) 获得（部分数据标注`*`代表数据来自原始论文），具体测试细节可参见 [OpenCompass](https://github.com/internLM/OpenCompass/) 中提供的配置文件。
- 评测数据会因 [OpenCompass](https://github.com/internLM/OpenCompass/) 的版本迭代而存在数值差异，请以 [OpenCompass](https://github.com/internLM/OpenCompass/) 最新版的评测结果为主。



**局限性：** 尽管在训练过程中我们非常注重模型的安全性，尽力促使模型输出符合伦理和法律要求的文本，但受限于模型大小以及概率生成范式，模型可能会产生各种不符合预期的输出，例如回复内容包含偏见、歧视等有害内容，请勿传播这些内容。由于传播不良信息导致的任何后果，本项目不承担责任。

</details>

## 使用案例

### 通过 Transformers 加载

通过以下的代码从 Transformers 加载 InternLM 模型 （可修改模型名称替换不同的模型）

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True).cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好！有什么我可以帮助你的吗？
>>> response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
>>> print(response)
当然可以！以下是三个管理时间的建议：
1. 制定计划：制定一个详细的计划，包括每天要完成的任务和活动。这将有助于您更好地组织时间，并确保您能够按时完成任务。
2. 优先级：将任务按照优先级排序，先完成最重要的任务。这将确保您能够在最短的时间内完成最重要的任务，从而节省时间。
3. 集中注意力：避免分心，集中注意力完成任务。关闭社交媒体和电子邮件通知，专注于任务，这将帮助您更快地完成任务，并减少错误的可能性。
```

### 通过 ModelScope 加载 

通过以下的代码从 ModelScope 加载 InternLM 模型 （可修改模型名称替换不同的模型）

```python
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b-v1_1', revision='v1.0.0')
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
streamlit run web_demo.py
```

效果如下

![效果](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

### 基于InternLM高性能部署

我们使用 [LMDeploy](https://github.com/InternLM/LMDeploy) 完成 InternLM 的一键部署。

1. 首先安装 LMDeploy:

  ```
  python3 -m pip install lmdeploy
  ```

2. 快速的部署命令如下：

  ```
  python3 -m lmdeploy.serve.turbomind.deploy InternLM-7B /path/to/internlm-7b/model hf
  ```

3. 在导出模型后，你可以直接通过如下命令启动服务一个服务并和部署后的模型对话

  ```
  python3 -m lmdeploy.serve.client {server_ip_addresss}:33337
  ```

[LMDeploy](https://github.com/InternLM/LMDeploy) 支持了 InternLM 部署的完整流程，请参考 [部署教程](https://github.com/InternLM/LMDeploy) 了解 InternLM 的更多部署细节。

## 微调&训练

### 预训练与微调使用教程

请参考[使用教程](./doc/usage.md)开始InternLM的安装、数据处理、预训练与微调。

### 转换为 Transformers 格式使用

通过 InternLM 进行训练的模型可以很轻松地转换为 HuggingFace Transformers 格式，方便与社区各种开源项目无缝对接。借助 `tools/transformers/convert2hf.py` 可以将训练保存的权重一键转换为 transformers 格式

```bash
python tools/transformers/convert2hf.py --src_folder origin_ckpt/ --tgt_folder hf_ckpt/ --tokenizer ./tools/V7_sft.model
```

转换之后可以通过以下的代码加载为 transformers

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

## 训练系统

### 系统结构

请参考[系统结构文档](./doc/structure.md)进一步了解。

### 训练性能

InternLM 深度整合了 Flash-Attention, Apex 等高性能模型算子，提高了训练效率。通过构建 Hybrid Zero 技术，实现计算和通信的高效重叠，大幅降低了训练过程中的跨节点通信流量。InternLM 支持 7B 模型从 8 卡扩展到 1024 卡，千卡规模下加速效率可高达 90%，训练吞吐超过 180TFLOPS，平均单卡每秒处理的 token 数量超过3600。下表为 InternLM 在不同配置下的扩展性测试数据：

| GPU Number         | 8   | 16  | 32  | 64  | 128  | 256  | 512  | 1024  |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| TGS | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| TFLOPS  | 193 | 191  | 188  | 188  | 187   | 185   | 186   | 184    |

TGS 代表平均每GPU每秒可以处理的 Token 数量。更多的性能测试数据可参考[训练性能文档](./doc/train_performance.md)进一步了解。

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
