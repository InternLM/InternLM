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
[🤔提交反馈](https://github.com/InternLM/InternLM/issues/new)|
[📜技术报告](https://arxiv.org/abs/2403.17297)<br>
[💬聊天应用](https://internlm-chat.intern-ai.org.cn/) |
[🔗API](https://internlm.intern-ai.org.cn/api/document) |
[🧩魔乐社区](https://modelers.cn/spaces/MindSpore-Lab/INTERNLM2-20B-PLAN)

[English](./README.md) |
[简体中文](./README_zh-CN.md)

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> 和 <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">微信社区</a>
</p>

## 简介

InternLM3，即书生·浦语大模型第3代，开源了80亿参数，面向通用使用与高阶推理的指令模型（InternLM3-8B-Instruct）。模型具备以下特点：

- **更低的代价取得更高的性能**:
  在推理、知识类任务上取得同量级最优性能，超过Llama3.1-8B和Qwen2.5-7B。值得关注的是InternLM3只用了4万亿词元进行训练，对比同级别模型训练成本节省75%以上。
- **深度思考能力**:
  InternLM3支持通过长思维链求解复杂推理任务的深度思考模式，同时还兼顾了用户体验更流畅的通用回复模式。

## 更新

\[2025.01.15\] 我们发布了 InternLM3-8B-Instruct 模型。可以在下方的 [模型库](#model-zoo) 进行下载，或者在 [model cards](./model_cards/) 中了解更多细节。

\[2024.08.01\] 我们发布了 InternLM2.5-1.8B、InternLM2.5-1.8B-Chat、InternLM2.5-20B 和 InternLM2.5-20B-Chat。可以在下方的 [模型库](#model-zoo) 进行下载，或者在 [model cards](./model_cards/) 中了解更多细节。

\[2024.07.19\] 我们发布了 1.8B、7B 和 20B 大小的 InternLM2-Reward 系列奖励模型。可以在下方的 [模型库](#model-zoo) 进行下载，或者在 [model cards](./model_cards/internlm2_reward.md) 中了解更多细节。

\[2024.06.30\] 我们发布了 InternLM2.5-7B、InternLM2.5-7B-Chat 和 InternLM2.5-7B-Chat-1M。可以在下方的 [模型库](#model-zoo) 进行下载，或者在 [model cards](./model_cards/) 中了解更多细节。

\[2024.03.26\] 我们发布了 InternLM2 的技术报告。 可以点击 [arXiv链接](https://arxiv.org/abs/2403.17297) 来了解更多细节。

\[2024.01.31\] 我们发布了 InternLM2-1.8B，以及相关的对话模型。该模型在保持领先性能的情况下，提供了更低廉的部署方案。

\[2024.01.23\] 我们发布了 InternLM2-Math-7B 和 InternLM2-Math-20B 以及相关的对话模型。InternLM-Math以较小的尺寸超过了ChatGPT的表现。可以点击[InternLM-Math](https://github.com/InternLM/internlm-math)进行下载，并了解详情。

\[2024.01.17\] 我们发布了 InternLM2-7B 和 InternLM2-20B 以及相关的对话模型，InternLM2 在数理、代码、对话、创作等各方面能力都获得了长足进步，综合性能达到开源模型的领先水平。可以点击[下面的模型库](#model-zoo)进行下载或者[查看模型文档](./model_cards/)来了解更多细节.

\[2023.12.13\] 我们更新了 InternLM-7B-Chat 和 InternLM-20B-Chat 模型权重。通过改进微调数据和训练策略，新版对话模型生成的回复质量更高、语言风格更加多元。

\[2023.09.20\] InternLM-20B 已发布，包括基础版和对话版。

## Model Zoo

### InternLM3

| Model                     | Transformers                                             | ModelScope                                             | Modelers                                              | Release Date |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- | ------------ |
| **InternLM3-8B-Instruct** | [🤗internlm3_8B_instruct](https://huggingface.co/internlm/internlm3-8b-instruct) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm3_8b_instruct](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm3-8b-instruct/summary) | [![Open in Modelers](https://modelers.cn/assets/logo1-1bf58310.svg)](https://modelers.cn/models/Intern/internlm3-8b-instruct) | 2025-01-15   |

### InternLM2.5

<details>
    <summary>(click to expand)</summary>

| Model                      | Transformers(HF)                           | ModelScope(HF)                           | OpenXLab(HF)                           | OpenXLab(Origin)                           | Release Date |
| -------------------------- | ------------------------------------------ | ---------------------------------------- | -------------------------------------- | ------------------------------------------ | ------------ |
| **InternLM2.5-1.8B**       | [🤗internlm2_5-1_8b](https://huggingface.co/internlm/internlm2_5-1_8b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-1_8b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-original) | 2024-08-05   |
| **InternLM2.5-1.8B-Chat**  | [🤗internlm2_5-1_8b-chat](https://huggingface.co/internlm/internlm2_5-1_8b-chat) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-1_8b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-chat) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-chat-original) | 2024-08-05   |
| **InternLM2.5-7B**         | [🤗internlm2_5-7b](https://huggingface.co/internlm/internlm2_5-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-original) | 2024-07-03   |
| **InternLM2.5-7B-Chat**    | [🤗internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-chat) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-chat-original) | 2024-07-03   |
| **InternLM2.5-7B-Chat-1M** | [🤗internlm2_5-7b-chat-1m](https://huggingface.co/internlm/internlm2_5-7b-chat-1m) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-7b-chat-1m](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-chat-1m) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-chat-1m-original) | 2024-07-03   |
| **InternLM2.5-20B**        | [🤗internlm2_5-20b](https://huggingface.co/internlm/internlm2_5-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-original) | 2024-08-05   |
| **InternLM2.5-20B-Chat**   | [🤗internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-20b-chat/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-chat) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-chat-original) | 2024-08-05   |

**模型说明：**

目前 InternLM 2.5 系列发布了 1.8B、7B 和 20B 大小的模型。7B 为轻量级的研究和应用提供了一个轻便但性能不俗的模型，20B 模型的综合性能更为强劲，可以有效支持更加复杂的实用场景。每个规格不同模型关系如下所示：

1. **InternLM2.5**：经历了大规模预训练的基座模型，是我们推荐的在大部分应用中考虑选用的优秀基座。
2. **InternLM2.5-Chat**: 对话模型，在 InternLM2.5 基座上经历了有监督微调和 online RLHF。InternLM2.5-Chat 面向对话交互进行了优化，具有较好的指令遵循、共情聊天和调用工具等的能力，是我们推荐直接用于下游应用的模型。
3. **InternLM2.5-Chat-1M**: InternLM2.5-Chat-1M 支持一百万字超长上下文，并具有和 InternLM2.5-Chat 相当的综合性能表现。

**局限性：** 尽管在训练过程中我们非常注重模型的安全性，尽力促使模型输出符合伦理和法律要求的文本，但受限于模型大小以及概率生成范式，模型可能会产生各种不符合预期的输出，例如回复内容包含偏见、歧视等有害内容，请勿传播这些内容。由于传播不良信息导致的任何后果，本项目不承担责任。

**补充说明：** 上表中的 `HF` 表示对应模型为 HuggingFace 平台提供的 [transformers](https://github.com/huggingface/transformers) 框架格式；`Origin` 则表示对应模型为我们 InternLM 团队的 [InternEvo](https://github.com/InternLM/InternEvo) 框架格式。

</details>

### InternLM2-Reward

<details>
    <summary>(click to expand)</summary>

InternLM2-Reward 是基于 240 万个偏好样本进行训练的奖励模型，有 1.8B、7B 和 20B 大小可供选择。这些模型被用于 InternLM 对话模型的 PPO 训练过程。请参考 [model cards](./model_cards/internlm2_reward.md) 了解更多细节。

| Model                     | RewardBench Score | Transformers(HF)                                   | ModelScope(HF)                                    | OpenXLab(HF)                                    | Release Date |
| ------------------------- | ----------------- | -------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------- | ------------ |
| **InternLM2-1.8B-Reward** | 80.6              | [🤗internlm2-1_8b-reward](https://huggingface.co/internlm/internlm2-1_8b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-1_8b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-1_8b-reward) | 2024-07-19   |
| **InternLM2-7B-Reward**   | 86.6              | [🤗internlm2-7b-reward](https://huggingface.co/internlm/internlm2-7b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-7b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b-reward) | 2024-07-19   |
| **InternLM2-20B-Reward**  | 89.5              | [🤗internlm2-20b-reward](https://huggingface.co/internlm/internlm2-20b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-20b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b-reward) | 2024-07-19   |

</details>

### InternLM2

<details>
    <summary>(click to expand)</summary>

我们上一代的模型，在长上下文处理、推理和编码方面具有优秀的性能。请参考 [model cards](./model_cards/) 了解更多细节。

| Model                       | Transformers(HF)                          | ModelScope(HF)                           | OpenXLab(HF)                           | OpenXLab(Origin)                           | Release Date |
| --------------------------- | ----------------------------------------- | ---------------------------------------- | -------------------------------------- | ------------------------------------------ | ------------ |
| **InternLM2-1.8B**          | [🤗internlm2-1.8b](https://huggingface.co/internlm/internlm2-1_8b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-1.8b](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-1.8b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-1.8b-original) | 2024-01-31   |
| **InternLM2-Chat-1.8B-SFT** | [🤗internlm2-chat-1.8b-sft](https://huggingface.co/internlm/internlm2-chat-1_8b-sft) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-1.8b-sft](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-1.8b-sft) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-1.8b-sft-original) | 2024-01-31   |
| **InternLM2-Chat-1.8B**     | [🤗internlm2-chat-1.8b](https://huggingface.co/internlm/internlm2-chat-1_8b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-1.8b](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-1.8b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-1.8b-original) | 2024-02-19   |
| **InternLM2-Base-7B**       | [🤗internlm2-base-7b](https://huggingface.co/internlm/internlm2-base-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b-original) | 2024-01-17   |
| **InternLM2-7B**            | [🤗internlm2-7b](https://huggingface.co/internlm/internlm2-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b-original) | 2024-01-17   |
| **InternLM2-Chat-7B-SFT**   | [🤗internlm2-chat-7b-sft](https://huggingface.co/internlm/internlm2-chat-7b-sft) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b-sft/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft-original) | 2024-01-17   |
| **InternLM2-Chat-7B**       | [🤗internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-original) | 2024-01-17   |
| **InternLM2-Base-20B**      | [🤗internlm2-base-20b](https://huggingface.co/internlm/internlm2-base-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b-original) | 2024-01-17   |
| **InternLM2-20B**           | [🤗internlm2-20b](https://huggingface.co/internlm/internlm2-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b-original) | 2024-01-17   |
| **InternLM2-Chat-20B-SFT**  | [🤗internlm2-chat-20b-sft](https://huggingface.co/internlm/internlm2-chat-20b-sft) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b-sft/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft-original) | 2024-01-17   |
| **InternLM2-Chat-20B**      | [🤗internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-original) | 2024-01-17   |

</details>

## 性能

我们使用开源评测工具 [OpenCompass](https://github.com/internLM/OpenCompass/) 从学科综合能力、语言能力、知识能力、推理能力、理解能力五大能力维度对InternLM开展全面评测，部分评测结果如下表所示，欢迎访问[ OpenCompass 榜单 ](https://rank.opencompass.org.cn)获取更多的评测结果。

| 评测集\\模型 |                                 | InternLM3-8B-Instruct | Qwen2.5-7B-Instruct | Llama3.1-8B-Instruct | GPT-4o-mini(close source) |
| ------------ | ------------------------------- | --------------------- | ------------------- | -------------------- | ------------------------- |
| General      | CMMLU(0-shot)                   | **83.1**              | 75.8                | 53.9                 | 66.0                      |
|              | MMLU(0-shot)                    | 76.6                  | **76.8**            | 71.8                 | 82.7                      |
|              | MMLU-Pro(0-shot)                | **57.6**              | 56.2                | 48.1                 | 64.1                      |
| Reasoning    | GPQA-Diamond(0-shot)            | **37.4**              | 33.3                | 24.2                 | 42.9                      |
|              | DROP(0-shot)                    | **83.1**              | 80.4                | 81.6                 | 85.2                      |
|              | HellaSwag(10-shot)              | **91.2**              | 85.3                | 76.7                 | 89.5                      |
|              | KOR-Bench(0-shot)               | **56.4**              | 44.6                | 47.7                 | 58.2                      |
| MATH         | MATH-500(0-shot)                | **83.0**\*            | 72.4                | 48.4                 | 74.0                      |
|              | AIME2024(0-shot)                | **20.0**\*            | 16.7                | 6.7                  | 13.3                      |
| Coding       | LiveCodeBench(2407-2409 Pass@1) | **17.8**              | 16.8                | 12.9                 | 21.8                      |
|              | HumanEval(Pass@1)               | 82.3                  | **85.4**            | 72.0                 | 86.6                      |
| Instrunction | IFEval(Prompt-Strict)           | **79.3**              | 71.7                | 75.2                 | 79.7                      |
| LongContext  | RULER(4-128K Average)           | 87.9                  | 81.4                | **88.5**             | 90.7                      |
| Chat         | AlpacaEval 2.0(LC WinRate)      | **51.1**              | 30.3                | 25.0                 | 50.7                      |
|              | WildBench(Raw Score)            | **33.1**              | 23.3                | 1.5                  | 40.3                      |
|              | MT-Bench-101(Score 1-10)        | **8.59**              | 8.49                | 8.37                 | 8.87                      |

- 以上评测结果基于 [OpenCompass](https://github.com/internLM/OpenCompass/) 获得(部分数据标注`*`代表使用深度思考模式进行评测)，具体测试细节可参见 [OpenCompass](https://github.com/internLM/OpenCompass/) 中提供的配置文件。
- 评测数据会因 [OpenCompass](https://github.com/internLM/OpenCompass/) 的版本迭代而存在数值差异，请以 [OpenCompass](https://github.com/internLM/OpenCompass/) 最新版的评测结果为主。

**局限性：** 尽管在训练过程中我们非常注重模型的安全性，尽力促使模型输出符合伦理和法律要求的文本，但受限于模型大小以及概率生成范式，模型可能会产生各种不符合预期的输出，例如回复内容包含偏见、歧视等有害内容，请勿传播这些内容。由于传播不良信息导致的任何后果，本项目不承担责任。

### 依赖

- Python >= 3.8
- PyTorch >= 1.12.0 (推荐 2.0.0 和更高版本)
- Transformers >= 4.38

## 使用案例

### 常规对话模式

#### Transformers 推理

通过以下的代码加载  InternLM3 8B Instruct 模型

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_dir = "internlm/internlm3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16)
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM3 8B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()
system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Please tell me five scenic spots in Shanghai"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
generated_ids = model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
]
prompt = tokenizer.batch_decode(tokenized_chat)[0]
print(prompt)
response = tokenizer.batch_decode(generated_ids)[0]
print(response)
```

#### LMDeploy 推理

LMDeploy 是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。

```bash
pip install lmdeploy
```

你可以使用以下 python 代码进行本地批量推理:

```python
import lmdeploy
model_dir = "internlm/internlm3-8b-instruct"
pipe = lmdeploy.pipeline(model_dir)
response = pipe(["Please tell me five scenic spots in Shanghai"])
print(response)
```

或者你可以使用以下命令启动兼容 OpenAI API 的服务:

```bash
lmdeploy serve api_server internlm/internlm3-8b-instruct --model-name internlm3-8b-instruct --server-port 23333
```

然后你可以向服务端发起一个聊天请求:

```bash
curl http://localhost:23333/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "internlm3-8b-instruct",
    "messages": [
    {"role": "user", "content": "介绍一下深度学习。"}
    ]
    }'
```

更多信息请查看 [LMDeploy 文档](https://lmdeploy.readthedocs.io/en/latest/)

#### Ollama 推理

安装ollama和拉取模型

```bash
# 安装 ollama
curl -fsSL https://ollama.com/install.sh | sh
# 拉取模型
ollama pull internlm/internlm3-8b-instruct
# 安装python库
pip install ollama
```

推理代码

```python
import ollama

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": "Please tell me five scenic spots in Shanghai"
    },
]

stream = ollama.chat(
    model='internlm/internlm3-8b-instruct',
    messages=messages,
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

#### vLLM 推理

参考[安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) 安装 vllm 最新代码

```python
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

推理代码

```python
from vllm import LLM, SamplingParams
llm = LLM(model="internlm/internlm3-8b-instruct")
sampling_params = SamplingParams(temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)
system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""
prompts = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": "Please tell me five scenic spots in Shanghai"
    },
]
outputs = llm.chat(prompts,
                   sampling_params=sampling_params,
                   use_tqdm=False)
print(outputs)
```

### 深度思考模式

#### 深度思考 Demo

<img src="https://github.com/InternLM/InternLM/blob/017ba7446d20ecc3b9ab8e7b66cc034500868ab4/assets/solve_puzzle.png?raw=true" width="400"/>

#### 深度思考 system prompt

```python
thinking_system_prompt = """You are an expert mathematician with extensive experience in mathematical competitions. You approach problems through systematic thinking and rigorous reasoning. When solving problems, follow these thought processes:
## Deep Understanding
Take time to fully comprehend the problem before attempting a solution. Consider:
- What is the real question being asked?
- What are the given conditions and what do they tell us?
- Are there any special restrictions or assumptions?
- Which information is crucial and which is supplementary?
## Multi-angle Analysis
Before solving, conduct thorough analysis:
- What mathematical concepts and properties are involved?
- Can you recall similar classic problems or solution methods?
- Would diagrams or tables help visualize the problem?
- Are there special cases that need separate consideration?
## Systematic Thinking
Plan your solution path:
- Propose multiple possible approaches
- Analyze the feasibility and merits of each method
- Choose the most appropriate method and explain why
- Break complex problems into smaller, manageable steps
## Rigorous Proof
During the solution process:
- Provide solid justification for each step
- Include detailed proofs for key conclusions
- Pay attention to logical connections
- Be vigilant about potential oversights
## Repeated Verification
After completing your solution:
- Verify your results satisfy all conditions
- Check for overlooked special cases
- Consider if the solution can be optimized or simplified
- Review your reasoning process
Remember:
1. Take time to think thoroughly rather than rushing to an answer
2. Rigorously prove each key conclusion
3. Keep an open mind and try different approaches
4. Summarize valuable problem-solving methods
5. Maintain healthy skepticism and verify multiple times
Your response should reflect deep mathematical understanding and precise logical thinking, making your solution path and reasoning clear to others.
When you're ready, present your complete solution with:
- Clear problem understanding
- Detailed solution process
- Key insights
- Thorough verification
Focus on clear, logical progression of ideas and thorough explanation of your mathematical reasoning. Provide answers in the same language as the user asking the question, repeat the final answer using a '\\boxed{}' without any units, you have [[8192]] tokens to complete the answer.
"""
```

#### Transformers 推理

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_dir = "internlm/internlm3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16)
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM3 8B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()
messages = [
    {"role": "system", "content": thinking_system_prompt},
    {"role": "user", "content": "已知函数\(f(x)=\mathrm{e}^{x}-ax - a^{3}\)。\n（1）当\(a = 1\)时，求曲线\(y = f(x)\)在点\((1,f(1))\)处的切线方程；\n（2）若\(f(x)\)有极小值，且极小值小于\(0\)，求\(a\)的取值范围。"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
generated_ids = model.generate(tokenized_chat, max_new_tokens=8192)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
]
prompt = tokenizer.batch_decode(tokenized_chat)[0]
print(prompt)
response = tokenizer.batch_decode(generated_ids)[0]
print(response)
```

#### LMDeploy 推理

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the MMRazor and MMDeploy teams.

```bash
pip install lmdeploy
```

You can run batch inference locally with the following python code:

```python
from lmdeploy import pipeline, GenerationConfig, ChatTemplateConfig
model_dir = "internlm/internlm3-8b-instruct"
chat_template_config = ChatTemplateConfig(model_name='internlm3')
pipe = pipeline(model_dir, chat_template_config=chat_template_config)
messages = [
        {"role": "system", "content": thinking_system_prompt},
        {"role": "user", "content": "已知函数\(f(x)=\mathrm{e}^{x}-ax - a^{3}\)。\n（1）当\(a = 1\)时，求曲线\(y = f(x)\)在点\((1,f(1))\)处的切线方程；\n（2）若\(f(x)\)有极小值，且极小值小于\(0\)，求\(a\)的取值范围。"},
]
response = pipe(messages, gen_config=GenerationConfig(max_new_tokens=2048))
print(response)
```

#### Ollama 推理

安装ollama和拉取模型

```bash
# 安装 ollama
curl -fsSL https://ollama.com/install.sh | sh
# 拉取模型
ollama pull internlm/internlm3-8b-instruct
# 安装python库
pip install ollama
```

推理代码

```python
import ollama

messages = [
    {
        "role": "system",
        "content": thinking_system_prompt,
    },
    {
        "role": "user",
        "content": "已知函数\(f(x)=\mathrm{e}^{x}-ax - a^{3}\)。\n（1）当\(a = 1\)时，求曲线\(y = f(x)\)在点\((1,f(1))\)处的切线方程；\n（2）若\(f(x)\)有极小值，且极小值小于\(0\)，求\(a\)的取值范围。"
    },
]

stream = ollama.chat(
    model='internlm/internlm3-8b-instruct',
    messages=messages,
    stream=True,
    options=dict(num_ctx=8192, num_predict=2048)
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

#### vLLM 推理

参考[安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) 安装 vllm 最新代码

```bash
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

推理代码

```python
from vllm import LLM, SamplingParams
llm = LLM(model="internlm/internlm3-8b-instruct")
sampling_params = SamplingParams(temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8, max_tokens=8192)
prompts = [
    {
        "role": "system",
        "content": thinking_system_prompt,
    },
    {
        "role": "user",
        "content": "已知函数\(f(x)=\mathrm{e}^{x}-ax - a^{3}\)。\n（1）当\(a = 1\)时，求曲线\(y = f(x)\)在点\((1,f(1))\)处的切线方程；\n（2）若\(f(x)\)有极小值，且极小值小于\(0\)，求\(a\)的取值范围。"
    },
]
outputs = llm.chat(prompts,
                   sampling_params=sampling_params,
                   use_tqdm=False)
print(outputs)
```

## 开源许可证

本仓库的代码和权重依照 Apache-2.0 协议开源。

## 引用

```
@misc{cai2024internlm2,
      title={InternLM2 Technical Report},
      author={Zheng Cai and Maosong Cao and Haojiong Chen and Kai Chen and Keyu Chen and Xin Chen and Xun Chen and Zehui Chen and Zhi Chen and Pei Chu and Xiaoyi Dong and Haodong Duan and Qi Fan and Zhaoye Fei and Yang Gao and Jiaye Ge and Chenya Gu and Yuzhe Gu and Tao Gui and Aijia Guo and Qipeng Guo and Conghui He and Yingfan Hu and Ting Huang and Tao Jiang and Penglong Jiao and Zhenjiang Jin and Zhikai Lei and Jiaxing Li and Jingwen Li and Linyang Li and Shuaibin Li and Wei Li and Yining Li and Hongwei Liu and Jiangning Liu and Jiawei Hong and Kaiwen Liu and Kuikun Liu and Xiaoran Liu and Chengqi Lv and Haijun Lv and Kai Lv and Li Ma and Runyuan Ma and Zerun Ma and Wenchang Ning and Linke Ouyang and Jiantao Qiu and Yuan Qu and Fukai Shang and Yunfan Shao and Demin Song and Zifan Song and Zhihao Sui and Peng Sun and Yu Sun and Huanze Tang and Bin Wang and Guoteng Wang and Jiaqi Wang and Jiayu Wang and Rui Wang and Yudong Wang and Ziyi Wang and Xingjian Wei and Qizhen Weng and Fan Wu and Yingtong Xiong and Chao Xu and Ruiliang Xu and Hang Yan and Yirong Yan and Xiaogui Yang and Haochen Ye and Huaiyuan Ying and Jia Yu and Jing Yu and Yuhang Zang and Chuyu Zhang and Li Zhang and Pan Zhang and Peng Zhang and Ruijie Zhang and Shuo Zhang and Songyang Zhang and Wenjian Zhang and Wenwei Zhang and Xingcheng Zhang and Xinyue Zhang and Hui Zhao and Qian Zhao and Xiaomeng Zhao and Fengzhe Zhou and Zaida Zhou and Jingming Zhuo and Yicheng Zou and Xipeng Qiu and Yu Qiao and Dahua Lin},
      year={2024},
      eprint={2403.17297},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
