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

InternLM2.5 系列模型在本仓库正式发布，具有如下特性：

- 卓越的推理性能：在数学推理方面取得了同量级模型最优精度，超越了 Llama3 和 Gemma2-9B。
- 有效支持百万字超长上下文：模型在 1 百万字长输入中几乎完美地实现长文“大海捞针”，而且在 LongBench 等长文任务中的表现也达到开源模型中的领先水平。 可以通过 [LMDeploy](./chat/lmdeploy_zh_cn.md) 尝试百万字超长上下文推理。更多内容和文档对话 demo 请查看[这里](./long_context/README_zh-CN.md)。
- 工具调用能力整体升级：InternLM2.5 支持从上百个网页搜集有效信息进行分析推理，相关实现将于近期开源到 [Lagent](https://github.com/InternLM/lagent/tree/main)。InternLM2.5 具有更强和更具有泛化性的指令理解、工具筛选与结果反思等能力，新版模型可以更可靠地支持复杂智能体的搭建，支持对工具进行有效的多轮调用，完成较复杂的任务。可以查看更多[样例](./agent/)。

## 更新

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

### InternLM2.5

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

### InternLM2-Reward

InternLM2-Reward 是基于 240 万个偏好样本进行训练的奖励模型，有 1.8B、7B 和 20B 大小可供选择。这些模型被用于 InternLM 对话模型的 PPO 训练过程。请参考 [model cards](./model_cards/internlm2_reward.md) 了解更多细节。

| Model                     | RewardBench Score | Transformers(HF)                                   | ModelScope(HF)                                    | OpenXLab(HF)                                    | Release Date |
| ------------------------- | ----------------- | -------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------- | ------------ |
| **InternLM2-1.8B-Reward** | 80.6              | [🤗internlm2-1_8b-reward](https://huggingface.co/internlm/internlm2-1_8b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-1_8b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-1_8b-reward) | 2024-07-19   |
| **InternLM2-7B-Reward**   | 86.6              | [🤗internlm2-7b-reward](https://huggingface.co/internlm/internlm2-7b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-7b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b-reward) | 2024-07-19   |
| **InternLM2-20B-Reward**  | 89.5              | [🤗internlm2-20b-reward](https://huggingface.co/internlm/internlm2-20b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-20b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b-reward) | 2024-07-19   |

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

我们使用开源评测工具 [OpenCompass](https://github.com/open-compass/opencompass) 在几个重要的基准测试中对 InternLM2.5 进行了评测。部分评测结果如下表所示。欢迎访问 [OpenCompass 排行榜](https://rank.opencompass.org.cn) 获取更多评测结果。

### 基座模型

| Benchmark      | InternLM2.5-7B | Llama3-8B | Yi-1.5-9B |
| -------------- | -------------- | --------- | --------- |
| MMLU (5-shot)  | **71.6**       | 66.4      | 71.6      |
| CMMLU (5-shot) | **79.1**       | 51.0      | 74.1      |
| BBH (3-shot)   | 70.1           | 59.7      | 71.1      |
| MATH (4-shot)  | **34.0**       | 16.4      | 31.9      |
| GSM8K (4-shot) | **74.8**       | 54.3      | 74.5      |
| GPQA (0-shot)  | **31.3**       | 31.3      | 27.8      |

### 对话模型

| Benchmark          | InternLM2.5-7B-Chat | Llama3-8B-Instruct | Gemma2-9B-IT | Yi-1.5-9B-Chat | GLM-4-9B-Chat | Qwen2-7B-Instruct |
| ------------------ | ------------------- | ------------------ | ------------ | -------------- | ------------- | ----------------- |
| MMLU (5-shot)      | **72.8**            | 68.4               | 70.9         | 71.0           | 71.4          | 70.8              |
| CMMLU (5-shot)     | 78.0                | 53.3               | 60.3         | 74.5           | 74.5          | 80.9              |
| BBH (3-shot CoT)   | **71.6**            | 54.4               | 68.2\*       | 69.6           | 69.6          | 65.0              |
| MATH (0-shot CoT)  | **60.1**            | 27.9               | 46.9         | 51.1           | 51.1          | 48.6              |
| GSM8K (0-shot CoT) | 86.0                | 72.9               | 88.9         | 80.1           | 85.3          | 82.9              |
| GPQA (0-shot)      | **38.4**            | 26.1               | 33.8         | 37.9           | 36.9          | 38.4              |

- 我们使用 `ppl` 对基座模型进行 MCQ 指标的评测。
- 评测结果来自 [OpenCompass](https://github.com/open-compass/opencompass) ，评测配置可以在 [OpenCompass](https://github.com/open-compass/opencompass) 提供的配置文件中找到。
- 由于 [OpenCompass](https://github.com/open-compass/opencompass) 的版本迭代，评测数据可能存在数值差异，因此请参考 [OpenCompass](https://github.com/open-compass/opencompass) 的最新评测结果。
- \* 表示从原论文中复制而来。

## 依赖

- Python >= 3.8
- PyTorch >= 1.12.0 (推荐 2.0.0 和更高版本)
- Transformers >= 4.38

## 使用案例

InternLM 支持众多知名的上下游项目，如 LLaMA-Factory、vLLM、llama.cpp 等。这种支持使得广大用户群体能够更高效、更方便地使用 InternLM 全系列模型。为方便使用，我们为部分生态系统项目提供了教程，访问[此处](./ecosystem/README_zh-CN.md)即可获取。

接下来我们展示使用 [Transformers](#import-from-transformers)，[ModelScope](#import-from-modelscope) 和 [Web demo](#dialogue) 进行推理。
对话模型采用了 [chatml 格式](./chat/chat_format.md) 来支持通用对话和智能体应用。
为了保障更好的使用效果，在用 [Transformers](#import-from-transformers) 或 [ModelScope](#import-from-modelscope) 进行推理前，请确保安装的 transformers 库版本满足以下要求：

```
transformers >= 4.38
```

### 通过 Transformers 加载

通过以下的代码从 Transformers 加载 InternLM2.5-7B-Chat 模型 （可修改模型名称替换不同的模型）

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
# 设置`torch_dtype=torch.float16`来将模型精度指定为torch.float16，否则可能会因为您的硬件原因造成显存不足的问题。
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", device_map="auto",trust_remote_code=True, torch_dtype=torch.float16)
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

通过以下的代码从 ModelScope 加载 InternLM2.5-7B-Chat 模型 （可修改模型名称替换不同的模型）

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2_5-7b-chat')
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
pip install transformers>=4.38
streamlit run ./chat/web_demo.py
```

## InternLM 高性能部署

我们使用 [LMDeploy](https://github.com/InternLM/LMDeploy) 完成 InternLM 的一键部署。

### 推理

通过 `pip install lmdeploy` 安装 LMDeploy 之后，只需 4 行代码，就可以实现离线批处理：

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2_5-7b-chat")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

为了减少内存占用，我们提供了4位量化模型 [internlm2_5-7b-chat-4bit](https://huggingface.co/internlm/internlm2_5-7b-chat-4bit)。可以按照如下方式推理该模型：

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2_5-7b-chat-4bit")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

此外，可以同步开启 8bit 或者 4bit KV 在线量化功能：

```python
from lmdeploy import pipeline, TurbomindEngineConfig
pipe = pipeline("internlm/internlm2_5-7b-chat-4bit",
                backend_config=TurbomindEngineConfig(quant_policy=8))
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

更多使用案例可参考[部署指南](./chat/lmdeploy.md)，详细的部署教程则可在[这里](https://github.com/InternLM/LMDeploy)找到。

### 1百万字超长上下文推理

激活 LMDeploy 的 Dynamic NTK 能力，可以轻松把 internlm2_5-7b-chat 外推到 200K 上下文。

注意: 1M 上下文需要 4xA100-80G。

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(
        rope_scaling_factor=2.5,
        session_len=1048576,  # 1M context length
        max_batch_size=1,
        cache_max_entry_count=0.7,
        tp=4)  # 4xA100-80G.
pipe = pipeline('internlm/internlm2_5-7b-chat-1m', backend_config=backend_config)
prompt = 'Use a long prompt to replace this sentence'
response = pipe(prompt)
print(response)
```

## 智能体

InternLM-2.5-Chat 模型有出色的工具调用性能并具有一定的零样本泛化能力。它支持从上百个网页中搜集信息并进行分析。更多样例可以参考  [agent 目录](./agent/).

## 微调&训练

请参考[微调教程](./finetune/)尝试续训或微调 InternLM2。

**注意：** 本项目中的全量训练功能已经迁移到了 [InternEvo](https://github.com/InternLM/InternEvo) 以便用户使用。InternEvo 提供了高效的预训练和微调基建用于训练 InternLM 系列模型。

## 评测

我们使用 [OpenCompass](https://github.com/open-compass/opencompass) 进行模型评估。在 InternLM2.5 中，我们主要标准客观评估、长文评估（大海捞针）、数据污染评估、智能体评估和主观评估。

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
@misc{cai2024internlm2,
      title={InternLM2 Technical Report},
      author={Zheng Cai and Maosong Cao and Haojiong Chen and Kai Chen and Keyu Chen and Xin Chen and Xun Chen and Zehui Chen and Zhi Chen and Pei Chu and Xiaoyi Dong and Haodong Duan and Qi Fan and Zhaoye Fei and Yang Gao and Jiaye Ge and Chenya Gu and Yuzhe Gu and Tao Gui and Aijia Guo and Qipeng Guo and Conghui He and Yingfan Hu and Ting Huang and Tao Jiang and Penglong Jiao and Zhenjiang Jin and Zhikai Lei and Jiaxing Li and Jingwen Li and Linyang Li and Shuaibin Li and Wei Li and Yining Li and Hongwei Liu and Jiangning Liu and Jiawei Hong and Kaiwen Liu and Kuikun Liu and Xiaoran Liu and Chengqi Lv and Haijun Lv and Kai Lv and Li Ma and Runyuan Ma and Zerun Ma and Wenchang Ning and Linke Ouyang and Jiantao Qiu and Yuan Qu and Fukai Shang and Yunfan Shao and Demin Song and Zifan Song and Zhihao Sui and Peng Sun and Yu Sun and Huanze Tang and Bin Wang and Guoteng Wang and Jiaqi Wang and Jiayu Wang and Rui Wang and Yudong Wang and Ziyi Wang and Xingjian Wei and Qizhen Weng and Fan Wu and Yingtong Xiong and Chao Xu and Ruiliang Xu and Hang Yan and Yirong Yan and Xiaogui Yang and Haochen Ye and Huaiyuan Ying and Jia Yu and Jing Yu and Yuhang Zang and Chuyu Zhang and Li Zhang and Pan Zhang and Peng Zhang and Ruijie Zhang and Shuo Zhang and Songyang Zhang and Wenjian Zhang and Wenwei Zhang and Xingcheng Zhang and Xinyue Zhang and Hui Zhao and Qian Zhao and Xiaomeng Zhao and Fengzhe Zhou and Zaida Zhou and Jingming Zhuo and Yicheng Zou and Xipeng Qiu and Yu Qiao and Dahua Lin},
      year={2024},
      eprint={2403.17297},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
