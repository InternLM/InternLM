# InternLM2-20B Model Card

## Introduction

The second generation of the InternLM model, InternLM2, includes models at two scales: 7B and 20B. For the convenience of users and researchers, we have open-sourced four versions of each scale of the model, which are:

- internlm2-base-20b: Foundation models with high quality and high adaptation flexibility, which serve as a good starting point for downstream deep adaptations.
- internlm2-20b (**recommended**): Further pretrain with general domain data and domain-enhanced corpus, obtaining state-of-the-art performance in evaluation with good language capability. InternLM2 models are recommended for consideration in most applications.
- internlm2-chat-20b-sft: Intermediate version of InternLM2-Chat that only undergoes supervised fine-tuning (SFT), based on the InternLM2-Base model. We release them to benefit research on alignment.
- internlm2-chat-20b (**recommended**): Further aligned on top of InternLM2-Chat-SFT through online RLHF. InternLM2-Chat exhibits better instruction following, chat experience, and function calling, which is recommended for downstream applications.

The base model of InternLM2 has the following technical features:

- Effective support for ultra-long contexts of up to 200,000 characters: The model nearly perfectly achieves "finding a needle in a haystack" in long inputs of 200,000 characters. It also leads among open-source models in performance on long-text tasks such as LongBench and L-Eval.
- Comprehensive performance enhancement: Compared to the previous generation model, it shows significant improvements in various capabilities, including reasoning, mathematics, and coding.

## Model Zoo

| Model                      | Transformers(HF)                           | ModelScope(HF)                           | OpenXLab(HF)                           | OpenXLab(Origin)                           | Release Date |
| -------------------------- | ------------------------------------------ | ---------------------------------------- | -------------------------------------- | ------------------------------------------ | ------------ |
| **InternLM2-Base-20B**     | [🤗internlm2-base-20b](https://huggingface.co/internlm/internlm2-base-20b) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2-base-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b-original) | 2024-01-17   |
| **InternLM2-20B**          | [🤗internlm2-20b](https://huggingface.co/internlm/internlm2-20b) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b-original) | 2024-01-17   |
| **InternLM2-Chat-20B-SFT** | [🤗internlm2-chat-20b-sft](https://huggingface.co/internlm/internlm2-chat-20b-sft) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b-sft/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft-original) | 2024-01-17   |
| **InternLM2-Chat-20B**     | [🤗internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-original) | 2024-01-17   |

- `HF` refers to the format used by HuggingFace in [transformers](https://github.com/huggingface/transformers), whereas `Origin` denotes the format adopted by the InternLM team in [InternEvo](https://github.com/InternLM/InternEvo).

## Performance Evaluation

We have evaluated InternLM2 on several important benchmarks using the open-source evaluation tool [OpenCompass](https://github.com/open-compass/opencompass). Some of the evaluation results are shown in the table below. You are welcome to visit the [OpenCompass Leaderboard](https://opencompass.org.cn/rank) for more evaluation results.

| Dataset\\Models | InternLM2-7B | InternLM2-Chat-7B | InternLM2-20B | InternLM2-Chat-20B | ChatGPT | GPT-4 |
| --------------- | ------------ | ----------------- | ------------- | ------------------ | ------- | ----- |
| MMLU            | 65.8         | 63.7              | 67.7          | 66.5               | 69.1    | 83.0  |
| AGIEval         | 49.9         | 47.2              | 53.0          | 50.3               | 39.9    | 55.1  |
| BBH             | 65.0         | 61.2              | 72.1          | 68.3               | 70.1    | 86.7  |
| GSM8K           | 70.8         | 70.7              | 76.1          | 79.6               | 78.2    | 91.4  |
| MATH            | 20.2         | 23.0              | 25.5          | 31.9               | 28.0    | 45.8  |
| HumanEval       | 43.3         | 59.8              | 48.8          | 67.1               | 73.2    | 74.4  |
| MBPP(Sanitized) | 51.8         | 51.4              | 63.0          | 65.8               | 78.9    | 79.0  |

- The evaluation results were obtained from [OpenCompass](https://github.com/open-compass/opencompass) , and evaluation configuration can be found in the configuration files provided by [OpenCompass](https://github.com/open-compass/opencompass).
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/open-compass/opencompass), so please refer to the latest evaluation results of [OpenCompass](https://github.com/open-compass/opencompass).
