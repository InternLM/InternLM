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

[📘使用文档](./doc/usage.md) |
[🛠️安装教程](./doc/install.md) |
[📊训练性能](./doc/train_performance.md) |
[👀模型库](#model-zoo) |
[🆕Update News](./CHANGE_LOG.md) |
[🤔Reporting Issues](https://github.com/InternLM/InternLM/issues/new)


[English](./README.md) |
[简体中文](./README-zh-Hans.md) 


</div>

## 简介
InternLM ，即书生·浦语大模型，包含面向实用场景的70亿参数基础模型与对话模型 （InternLM-7B）。模型具有以下特点：
- 使用上万亿高质量预料，建立模型超强知识体系；
- 支持8k语境窗口长度，实现更长输入与更强推理体验；
- 通用工具调用能力，支持用户灵活自助搭建流程；

提供了支持模型预训练的轻量级训练框架，无需安装大量依赖包，一套代码支持千卡预训练和单卡人类偏好对齐训练，同时实现了极致的性能优化，实现千卡训练下近90%加速效率。

## InternLM-7B

### 性能评测

我们使用开源评测工具 [OpenCompass](https://github.com/internLM/OpenCompass/) 从学科综合能力、语言能力、知识能力、推理能力、理解能力五大能力维度对InternLM开展全面评测，部分评测结果如下表所示，欢迎访问[ OpenCompass 榜单 ](https://opencompass.org.cn/rank)获取更多的评测结果。

| 数据集\模型           |  **InternLM-Chat-7B** |  **InternLM-7B**  |  LLaMA-7B | Baichuan-7B | ChatGLM2-6B | Alpaca-7B | Vicuna-7B |   
| -------------------- | --------------------- | ---------------- | --------- |  --------- | ------------ | --------- | ---------- |  
| C-Eval(Val)          |      53.2             |        53.4       | 24.2      | 42.7       |  50.9       |  28.9     | 31.2     |
| MMLU                 |      50.8             |       51.0        | 35.2*     |  41.5      |  46.0       |  39.7     | 47.3     |
| AGIEval              |      42.5             |       37.6        | 20.8      | 24.6       |  39.0       | 24.1      | 26.4     |
| CommonSenseQA        |      75.2             |      59.5         | 65.0      | 58.8       | 60.0        | 68.7      | 66.7     |
| BUSTM                |      74.3             |       50.6        | 48.5      | 51.3        | 55.0        | 48.8      | 62.5     |
| CLUEWSC              |      78.6             |      59.1         |  50.3     |  52.8     |  59.8     |   50.3    |  52.2     | 
| CommonSenseQA        |      75.2            |       59.5         | 60.0      |  58.8	   |  60.0     |   68.7    |  66.7     | 
| MATH                 |      6.4            |         7.1        |  2.8       | 3.0       | 6.6       |  2.2      | 2.8       |
| GSM8K                |      34.5           |        31.2        | 10.1       | 9.7       | 29.2      |  6.0      | 15.3  |
|  HumanEval           |      14.0           |        10.4        |   14.0     | 9.2       | 9.2       | 9.2       | 11.0  |
| RACE(High)           |      76.3           |        57.4        | 46.9*      | 28.1      | 66.3      | 40.7      | 54.0  | 

- 以上评测结果基于 [OpenCompass 20230706](https://github.com/internLM/OpenCompass/) 获得（部分数据标注`*`代表数据来自原始论文），具体测试细节可参见 [OpenCompass](https://github.com/internLM/OpenCompass/) 中提供的配置文件。
- 评测数据会因 [OpenCompass](https://github.com/internLM/OpenCompass/) 的版本迭代而存在数值差异，请以 [OpenCompass](https://github.com/internLM/OpenCompass/) 最新版的评测结果为主。

### Model Zoo
当前通过 InternLM 训练的 InternLM 7B 和 InternLM 7B Chat 已经开源，我们提供两种格式的模型权重以供使用。除了使用 Transformers 格式加载模型之外，还可以通过 InternLM 加载以下格式的权重直接进行继续预训练或人类偏好对齐训练

| 模型                 | InternLM 格式权重下载地址                                                                                                                      | Transformers 格式权重下载地址                    |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **InternLM 7B**      | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-7b) | [🤗internlm/intern-7b](https://huggingface.co/internlm/internlm-7b) |
| **InternLM Chat 7B** | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b) | [🤗internlm/intern-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)

**局限性：** 尽管在训练过程中我们非常注重模型的安全性，尽力促使模型输出符合伦理和法律要求的文本，但受限于模型大小以及概率生成范式，模型可能会产生各种不符合预期的输出，例如回复内容包含偏见、歧视等有害内容，请勿传播这些内容。由于传播不良信息导致的任何后果，本项目不承担责任。

### 通过 Transformers 加载
通过以下的代码加载 InternLM 7B Chat 模型
```python
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True, device='cuda')
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
通过 InternLM 进行训练的模型可以很轻松地转换为 HuggingFace Transformers 格式，方便与社区各种开源项目无缝对接。借助 `tools/convert2hf.py` 可以将训练保存的权重一键转换为 transformers 格式
```bash
python convert2hf.py --src_folder origin_ckpt/ --tgt_folder hf_ckpt/ --tokenizer tokenizes/tokenizer.model
```
转换之后可以通过以下的代码加载为 transformers 
```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, device='cuda')
```


## 训练系统

### 系统结构
请参考[系统结构文档](./doc/structure.md)进一步了解。

### 训练性能

InternLM 深度整合了 Flash-Attention, Apex 等高性能模型算子，提高了训练效率。通过构建 Hybrid Zero 技术，实现计算和通信的高效重叠，大幅降低了训练过程中的跨节点通信流量。InternLM 支持 7B 模型从 8 卡扩展到 1024 卡，千卡规模下加速效率可高达 90%，训练吞吐超过 180TFLOPS，平均单卡每秒处理的 token 数量超过3600。下表为 InternLM 在不同配置下的扩展性测试数据：

| GPU数量         | 8  | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| TKS | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| TFLOPS  | 192 | 192  | 186  | 186  | 185   | 185   | 186   | 182    |

TKS 代表平均每GPU每秒可以处理的 Token 数量。更多的性能测试数据可参考[训练性能文档](./doc/train_performance.md)进一步了解。


## 贡献

我们感谢所有的贡献者为改进和提升 InternLM 所作出的努力。非常欢迎社区用户能参与进项目中来。请参考贡献指南来了解参与项目贡献的相关指引。

## 致谢

InternLM 代码库是一款由上海人工智能实验室和来自不同高校、企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望这个工具箱和基准测试可以为社区提供灵活高效的代码工具，供用户微调 InternLM 并开发自己的新模型，从而不断为开源社区提供贡献。特别鸣谢flash-attention (https://github.com/HazyResearch/flash-attention) 与 ColossalAI (https://github.com/hpcaitech/ColossalAI) 两项开源项目。

## 开源许可证
 
本仓库的代码依照 Apache-2.0 协议开源。InternLM 权重对学术研究完全开放，在获得官方的书面许可后，亦允许商业使用。申请商用许可与合作请联系 internlm@pjlab.org.cn。
