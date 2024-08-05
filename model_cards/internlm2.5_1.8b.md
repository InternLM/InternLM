# InternLM2.5-1.8B Model Card

## Introduction

InternLM2.5, the 2.5th generation InternLM, has open-sourced a 1.8 billion parameter base model and a chat model tailored for practical scenarios. For the convenience of users and researchers, we have open-sourced two versions of each scale of the model, which are:

- InternLM2.5-1.8B: Further pretrain with general domain data and domain-enhanced corpus, obtaining state-of-the-art performance in evaluation with good language capability. InternLM2.5 models are recommended for consideration in most applications.
- InternLM2.5-chat-1.8B: Further aligned on top of InternLM2.5 through supervised fine-tuning (SFT) and online RLHF. InternLM2.5-Chat exhibits better instruction following, chat experience, and function calling, which is recommended for downstream applications.

The model has the following characteristics:

- **Outstanding reasoning capability**: State-of-the-art performance on Math reasoning, surpassing models like MiniCPM-2 and Qwen2-1.5B.

- **Stronger tool use**: InternLM2.5 supports gathering information from more than 100 web pages, corresponding implementation has be released in [MindSearch](https://github.com/InternLM/MindSearch). InternLM2.5 has better tool utilization-related capabilities in instruction following, tool selection and reflection. See [examples](https://github.com/InternLM/InternLM/blob/main/agent/lagent.md).

## Model Zoo

| Model                     | Transformers(HF)                           | ModelScope(HF)                           | OpenXLab(HF)                           | OpenXLab(Origin)                            | Release Date |
| ------------------------- | ------------------------------------------ | ---------------------------------------- | -------------------------------------- | ------------------------------------------- | ------------ |
| **InternLM2.5-1.8B**      | [🤗internlm2_5-1_8b](https://huggingface.co/internlm/internlm2_5-1_8b) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2_5-1_8b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-original) | 2024-08-05   |
| **InternLM2.5-1.8B-Chat** | [🤗internlm2_5-1_8b-chat](https://huggingface.co/internlm/internlm2_5-1_8b-chat) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2_5-1_8b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-chat) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-1_8b-chat-original) | 2024-08-05   |

- `HF` refers to the format used by HuggingFace in [transformers](https://github.com/huggingface/transformers), whereas `Origin` denotes the format adopted by the InternLM team in [InternEvo](https://github.com/InternLM/InternEvo).

## Performance Evaluation

We have evaluated InternLM2.5 on several important benchmarks using the open-source evaluation tool [OpenCompass](https://github.com/open-compass/opencompass). Some of the evaluation results are shown in the table below. You are welcome to visit the [OpenCompass Leaderboard](https://opencompass.org.cn/rank) for more evaluation results.

### Base Model

| Benchmark | InternLM2.5-1.8B | InternLM2-1.8B | Qwen2-1.5B |
| --------- | ---------------- | -------------- | ---------- |
| MMLU      | 53.52            | 45.99          | 57.45      |
| CMMLU     | 65.44            | 45.27          | 70.58      |
| BBH       | 41.16            | 36.03          | 35.75      |
| MATH      | 27.28            | 9.42           | 24.38      |
| HUMANEVAL | 35.98            | 30.49          | 34.15      |
| GPQA      | 24.24            | 24.24          | 31.82      |

### Chat Model

| Benchmark         | InternLM2.5-1.8B-Chat | MiniCPM-2 | Qwen2-1.5B-Instruct |
| ----------------- | --------------------- | --------- | ------------------- |
| MMLU (5-shot)     | 50.7                  | 54.2      | 55.7                |
| CMMLU (5-shot)    | 62.2                  | 50.6      | 65.2                |
| BBH (3-shot CoT)  | **41.9**              | 41.5      | 36.5                |
| MATH (0-shot CoT) | **40.2**              | 15.5      | 21.4                |
| GPQA (0-shot)     | **27.8**              | 23.7      | 27.3                |

- We use `ppl` for the MCQ evaluation on base model.
- The evaluation results were obtained from [OpenCompass](https://github.com/open-compass/opencompass) , and evaluation configuration can be found in the configuration files provided by [OpenCompass](https://github.com/open-compass/opencompass).
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/open-compass/opencompass), so please refer to the latest evaluation results of [OpenCompass](https://github.com/open-compass/opencompass).
- \* means the result is copied from the original paper.
