# InternLM

<div align="center">

<img src="./assets/logo.svg" width="200"/>
  <div> </div>
  <div align="center">
    <b><font size="5">InternLM</font></b>
    <sup>
      <a href="https://internlm.intern-ai.org.cn/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    <div> </div>
  </div>

[![license](./assets/license.svg)](./LICENSE)
[![evaluation](./assets/compass_support.svg)](https://github.com/internLM/OpenCompass/)

<!-- [![Documentation Status](https://readthedocs.org/projects/internlm/badge/?version=latest)](https://internlm.readthedocs.io/zh_CN/latest/?badge=latest) -->

[📘Commercial Application](#license) |
[🤗HuggingFace](https://huggingface.co/internlm) |
[🆕Update News](#news) |
[🤔Reporting Issues](https://github.com/InternLM/InternLM/issues/new) |
[📜Technical Report](https://arxiv.org/abs/2403.17297)<br>
[💬Chat Web](https://internlm-chat.intern-ai.org.cn/) |
[🔗API](https://internlm.intern-ai.org.cn/api/document) |
[🧩Modelers](https://modelers.cn/spaces/MindSpore-Lab/INTERNLM2-20B-PLAN)

[English](./README.md) |
[简体中文](./README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">WeChat</a>
</p>

## Introduction

InternLM3 has open-sourced an 8-billion parameter instruction model, InternLM3-8B-Instruct, designed for general-purpose usage and advanced reasoning. This model has the following characteristics:

- **Enhanced performance at reduced cost**:
  State-of-the-art performance on reasoning and knowledge-intensive tasks surpass models like Llama3.1-8B and Qwen2.5-7B. Remarkably, InternLM3 is trained on only 4 trillion high-quality tokens, saving more than 75% of the training cost compared to other LLMs of similar scale.
- **Deep thinking capability**:
  InternLM3 supports both the deep thinking mode for solving complicated reasoning tasks via the long chain-of-thought and the normal response mode for fluent user interactions.

## News

\[2025.01.15\] We release InternLM3-8B-Instruct, See [model zoo below](#model-zoo) for download or [model cards](./model_cards/) for more details.

\[2024.08.01\] We release InternLM2.5-1.8B, InternLM2.5-1.8B-Chat, InternLM2.5-20B and InternLM2.5-20B-Chat. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/) for more details.

\[2024.07.19\] We release the InternLM2-Reward series of reward models in 1.8B, 7B and 20B sizes. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/internlm2_reward.md) for more details.

\[2024.07.03\] We release InternLM2.5-7B, InternLM2.5-7B-Chat and InternLM2.5-7B-Chat-1M. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/) for more details.

\[2024.03.26\] We release InternLM2 technical report. See [arXiv](https://arxiv.org/abs/2403.17297) for details.

\[2024.01.31\] We release InternLM2-1.8B, along with the associated chat model. They provide a cheaper deployment option while maintaining leading performance.

\[2024.01.23\] We release InternLM2-Math-7B and InternLM2-Math-20B with pretraining and SFT checkpoints. They surpass ChatGPT with small sizes. See [InternLM-Math](https://github.com/InternLM/internlm-math) for details and download.

\[2024.01.17\] We release InternLM2-7B and InternLM2-20B and their corresponding chat models with stronger capabilities in all dimensions. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/) for more details.

\[2023.12.13\] InternLM-7B-Chat and InternLM-20B-Chat checkpoints are updated. With an improved finetuning strategy, the new chat models can generate higher quality responses with greater stylistic diversity.

\[2023.09.20\] InternLM-20B is released with base and chat versions.

## Model Zoo

### InternLM3

| Model                     | Transformers(HF)                                         | ModelScope(HF)                                         | Modelers(HF)                                          | Release Date |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- | ------------ |
| **InternLM3-8B-Instruct** | [🤗internlm3_8B_instruct](https://huggingface.co/internlm/internlm3-8b-instruct) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm3_8b_instruct](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm3-8b-instruct/summary) | [![Open in Modelers](<>)](https://modelers.cn/models/Intern/internlm3-8b-instruct) | 2025-01-15   |

### InternLM2.5

<details>
    <summary>(click to expand)</summary>

| Model                      | Transformers(HF)                           | ModelScope(HF)                           | OpenXLab(HF)                           | OpenXLab(Origin)                           | Release Date |
| -------------------------- | ------------------------------------------ | ---------------------------------------- | -------------------------------------- | ------------------------------------------ | ------------ |
| **InternLM2.5-1.8B**       | [🤗internlm2_5-1_8b](https://huggingface.co/internlm/internlm2_5-1_8b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-1_8b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-original) | 2024-08-05   |
| **InternLM2.5-1.8B-Chat**  | [🤗internlm2_5-1_8b-chat](https://huggingface.co/internlm/internlm2_5-1_8b-chat) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-1_8b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-chat) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-chat-original) | 2024-08-05   |
| **InternLM2.5-7B**         | [🤗internlm2_5-7b](https://huggingface.co/internlm/internlm2_5-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-original) | 2024-07-03   |
| **InternLM2.5-7B-Chat**    | [🤗internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-chat) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-chat-original) | 2024-07-03   |
| **InternLM2.5-7B-Chat-1M** | [🤗internlm2_5-7b-chat-1m](https://huggingface.co/internlm/internlm2_5-7b-chat-1m) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-7b-chat-1m](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-chat-1m) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-7b-chat-1m-original) | 2024-07-03   |
| **InternLM2.5-20B**        | [🤗internlm2_5-20b](https://huggingface.co/internlm/internlm2_5-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-original) | 2024-08-05   |
| **InternLM2.5-20B-Chat**   | [🤗internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2_5-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-20b-chat/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-chat) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-chat-original) | 2024-08-05   |

**Notes:**

The release of InternLM2.5 series contains 1.8B, 7B, and 20B versions. 7B models are efficient for research and application and 20B models are more powerful and can support more complex scenarios. The relation of these models are shown as follows.

1. **InternLM2.5**: Foundation models pre-trained on large-scale corpus. InternLM2.5 models are recommended for consideration in most applications.
2. **InternLM2.5-Chat**: The Chat model that undergoes supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF), based on the InternLM2.5 model. InternLM2.5-Chat is optimized for instruction following, chat experience, and function call, which is recommended for downstream applications.
3. **InternLM2.5-Chat-1M**: InternLM2.5-Chat-1M supports 1M long-context with compatible performance as InternLM2.5-Chat.

**Limitations:** Although we have made efforts to ensure the safety of the model during the training process and to encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the dissemination of harmful information.

**Supplements:** `HF` refers to the format used by HuggingFace in [transformers](https://github.com/huggingface/transformers), whereas `Origin` denotes the format adopted by the InternLM team in [InternEvo](https://github.com/InternLM/InternEvo).

</details>

### InternLM2-Reward

<details>
    <summary>(click to expand)</summary>

InternLM2-Reward is a series of reward models, trained on 2.4 million preference samples, available in 1.8B, 7B, and 20B sizes. These model were applied to the PPO training process of our chat models. See [model cards](./model_cards/internlm2_reward.md) for more details.

| Model                     | RewardBench Score | Transformers(HF)                                   | ModelScope(HF)                                    | OpenXLab(HF)                                    | Release Date |
| ------------------------- | ----------------- | -------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------- | ------------ |
| **InternLM2-1.8B-Reward** | 80.6              | [🤗internlm2-1_8b-reward](https://huggingface.co/internlm/internlm2-1_8b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-1_8b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-1_8b-reward) | 2024-07-19   |
| **InternLM2-7B-Reward**   | 86.6              | [🤗internlm2-7b-reward](https://huggingface.co/internlm/internlm2-7b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-7b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b-reward) | 2024-07-19   |
| **InternLM2-20B-Reward**  | 89.5              | [🤗internlm2-20b-reward](https://huggingface.co/internlm/internlm2-20b-reward) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-20b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b-reward) | 2024-07-19   |

</details>

### InternLM2

<details>
    <summary>(click to expand)</summary>

Our previous generation models with advanced capabilities in long-context processing, reasoning, and coding. See [model cards](./model_cards/) for more details.

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

## Performance

We conducted a comprehensive evaluation of InternLM using the open-source evaluation tool [OpenCompass](https://github.com/internLM/OpenCompass/). The evaluation covered five dimensions of capabilities: disciplinary competence, language competence, knowledge competence, inference competence, and comprehension competence. Here are some of the evaluation results, and you can visit the [OpenCompass leaderboard](https://rank.opencompass.org.cn) for more evaluation results.

| Benchmark    |                                 | InternLM3-8B-Instruct | Qwen2.5-7B-Instruct | Llama3.1-8B-Instruct | GPT-4o-mini(close source) |
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
| Long Context | RULER(4-128K Average)           | 87.9                  | 81.4                | **88.5**             | 90.7                      |
| Chat         | AlpacaEval 2.0(LC WinRate)      | **51.1**              | 30.3                | 25.0                 | 50.7                      |
|              | WildBench(Raw Score)            | **33.1**              | 23.3                | 1.5                  | 40.3                      |
|              | MT-Bench-101(Score 1-10)        | **8.59**              | 8.49                | 8.37                 | 8.87                      |

- The evaluation results were obtained from [OpenCompass](https://github.com/internLM/OpenCompass/) (some data marked with \*, which means evaluating with Thinking Mode), and evaluation configuration can be found in the configuration files provided by [OpenCompass](https://github.com/internLM/OpenCompass/).
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/internLM/OpenCompass/), so please refer to the latest evaluation results of [OpenCompass](https://github.com/internLM/OpenCompass/).
  **Limitations:** Although we have made efforts to ensure the safety of the model during the training process and to encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the dissemination of harmful information.

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0 (2.0.0 and above are recommended)
- Transformers >= 4.38

## Usages

InternLM supports a diverse range of well-known upstream and downstream projects, such as LLaMA-Factory, vLLM, llama.cpp, and more. This support enables a broad spectrum of users to utilize the InternLM series models more efficiently and conveniently. Tutorials for selected ecosystem projects are available [here](./ecosystem/README.md) for your convenience.

In the following chapters, we will focus on the usages with [Transformers](#import-from-transformers), [ModelScope](#import-from-modelscope), and [Web demos](#dialogue).
The chat models adopt [chatml format](./chat/chat_format.md) to support both chat and agent applications.
To ensure a better usage effect, please make sure that the installed transformers library version meets the following requirements before performing inference with [Transformers](#import-from-transformers) or [ModelScope](#import-from-modelscope):

```
transformers >= 4.48
```

### Import from Transformers

To load the InternLM3-8B-Instruct model using Transformers, use the following code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm3-8b-instruct", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm3-8b-instruct", trust_remote_code=True, torch_dtype=torch.float16)
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM3 8B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained("internlm/internlm3-8b-instruct", device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained("internlm/internlm3-8b-instruct", device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()

messages = [
    {"role": "system", "content": "You are an AI assistant whose name is InternLM."},
    {"role": "user", "content": "Please tell me five scenic spots in Shanghai"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

generated_ids = model.generate(tokenized_chat, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
]
response = tokenizer.batch_decode(generated_ids)[0]
```

### Import from ModelScope

To load the InternLM3-8B-Instruct model using ModelScope, use the following code:

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm3-8b-instruct')
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16)
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM3 8B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()

messages = [
    {"role": "system", "content": "You are an AI assistant whose name is InternLM."},
    {"role": "user", "content": "Please tell me five scenic spots in Shanghai"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

generated_ids = model.generate(tokenized_chat, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
]
response = tokenizer.batch_decode(generated_ids)[0]
```

### Dialogue

You can interact with the InternLM3-8B-Instruct model through a frontend interface by running the following code:

```bash
pip install streamlit
pip install transformers>=4.48
streamlit run ./chat/web_demo.py
```

## Deployment by LMDeploy

We use [LMDeploy](https://github.com/InternLM/LMDeploy) for fast deployment of InternLM.

### Inference

With only 4 lines of codes, you can perform [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat) inference after `pip install lmdeploy`.

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2_5-7b-chat")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

To reduce the memory footprint, we offers 4-bit quantized model [internlm2_5-7b-chat-4bit](https://huggingface.co/internlm/internlm2_5-7b-chat-4bit), with which the inference can be conducted as follows:

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2_5-7b-chat-4bit")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

Moreover, you can independently activate the 8bit/4bit KV cache feature:

```python
from lmdeploy import pipeline, TurbomindEngineConfig
pipe = pipeline("internlm/internlm2_5-7b-chat-4bit",
                backend_config=TurbomindEngineConfig(quant_policy=8))
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

Please refer to the [guidance](./chat/lmdeploy.md) for more usages about model deployment. For additional deployment tutorials, feel free to explore [here](https://github.com/InternLM/LMDeploy).

### 1M-long-context Inference

By enabling the Dynamic NTK feature of LMDeploy, you can acquire the long-context inference power.

Note: 1M context length requires 4xA100-80G.

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

## Agent

InternLM2.5-Chat models have excellent tool utilization capabilities and can work with function calls in a zero-shot manner. It also supports to conduct analysis by collecting information from more than 100 web pages. See more examples in [agent section](./agent/).

## Fine-tuning

Please refer to [finetune docs](./finetune/) for fine-tuning with InternLM.

**Note:** We have migrated the whole training functionality in this project to [InternEvo](https://github.com/InternLM/InternEvo) for easier user experience, which provides efficient pre-training and fine-tuning infra for training InternLM.

## Evaluation

We utilize [OpenCompass](https://github.com/open-compass/opencompass) for model evaluation. In InternLM2.5, we primarily focus on standard objective evaluation, long-context evaluation (needle in a haystack), data contamination assessment, agent evaluation, and subjective evaluation.

### Objective Evaluation

To evaluate the InternLM model, please follow the guidelines in the [OpenCompass tutorial](https://opencompass.readthedocs.io/en/latest/get_started/installation.html). Typically, we use `ppl` for multiple-choice questions on the **Base** model and `gen` for all questions on the **Chat** model.

### Long-Context Evaluation (Needle in a Haystack)

For the `Needle in a Haystack` evaluation, refer to the tutorial provided in the [documentation](https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/needleinahaystack_eval.md). Feel free to try it out.

### Data Contamination Assessment

To learn more about data contamination assessment, please check the [contamination eval](https://opencompass.readthedocs.io/en/latest/advanced_guides/contamination_eval.html).

### Agent Evaluation

- To evaluate tool utilization, please refer to [T-Eval](https://github.com/open-compass/T-Eval).
- For code interpreter evaluation, use the [Math Agent Evaluation](agent/README.md) provided in the repository.

### Subjective Evaluation

- Please follow the [tutorial](https://opencompass.readthedocs.io/en/latest/advanced_guides/subjective_evaluation.html) for subjective evaluation.

## Contribution

We appreciate all the contributors for their efforts to improve and enhance InternLM. Community users are highly encouraged to participate in the project. Please refer to the contribution guidelines for instructions on how to contribute to the project.

## License

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [application form (English)](https://wj.qq.com/s2/12727483/5dba/)/[申请表（中文）](https://wj.qq.com/s2/12725412/f7c1/). For other questions or collaborations, please contact <internlm@pjlab.org.cn>.

## Citation

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
