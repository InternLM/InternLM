# InternLM2.5-20B Model Card

## Introduction

InternLM2.5, the 2.5th generation InternLM, has open-sourced a 20 billion parameter base model and a chat model tailored for practical scenarios. For the convenience of users and researchers, we have open-sourced two versions of each scale of the model, which are:

- InternLM2.5-20B: Further pretrain with general domain data and domain-enhanced corpus, obtaining state-of-the-art performance in evaluation with good language capability. InternLM2.5 models are recommended for consideration in most applications.
- InternLM2.5-chat-20B: Further aligned on top of InternLM2.5 through supervised fine-tuning (SFT) and online RLHF. InternLM2.5-Chat exhibits better instruction following, chat experience, and function calling, which is recommended for downstream applications.

The model has the following characteristics:

- **Outstanding reasoning capability**: State-of-the-art performance on Math reasoning, surpassing models like Llama3 and Gemma2-27B.

- **Stronger tool use**: InternLM2.5 supports gathering information from more than 100 web pages, corresponding implementation has be released in [MindSearch](https://github.com/InternLM/MindSearch). InternLM2.5 has better tool utilization-related capabilities in instruction following, tool selection and reflection. See [examples](https://github.com/InternLM/InternLM/blob/main/agent/lagent.md).

## Model Zoo

| Model                    | Transformers(HF)                           | ModelScope(HF)                           | OpenXLab(HF)                            | OpenXLab(Origin)                            | Release Date |
| ------------------------ | ------------------------------------------ | ---------------------------------------- | --------------------------------------- | ------------------------------------------- | ------------ |
| **InternLM2.5-20B**      | [🤗internlm2_5-20b](https://huggingface.co/internlm/internlm2_5-20b) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2_5-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-original) | 2024-08-05   |
| **InternLM2.5-20B-Chat** | [🤗internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2_5-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-20b-chat/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-chat) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2_5-20b-chat-original) | 2024-08-05   |

- `HF` refers to the format used by HuggingFace in [transformers](https://github.com/huggingface/transformers), whereas `Origin` denotes the format adopted by the InternLM team in [InternEvo](https://github.com/InternLM/InternEvo).

## Performance Evaluation

We have evaluated InternLM2.5 on several important benchmarks using the open-source evaluation tool [OpenCompass](https://github.com/open-compass/opencompass). Some of the evaluation results are shown in the table below. You are welcome to visit the [OpenCompass Leaderboard](https://opencompass.org.cn/rank) for more evaluation results.

### Base Model

| Benchmark | InternLM2.5-20B | InternLM2-20B |
| --------- | --------------- | ------------- |
| MMLU      | 74.25           | 67.58         |
| CMMLU     | 82.22           | 68.29         |
| BBH       | 77.82           | 71.36         |
| MATH      | 48              | 32.66         |
| HUMANEVAL | 71.95           | 51.22         |
| GPQA      | 37.88           | 31.31         |

### Chat Model

| Benchmark         | InternLM2.5-20B-Chat | Gemma2-27B-IT |
| ----------------- | -------------------- | ------------- |
| MMLU (5-shot)     | 73.5                 | 75.0          |
| CMMLU (5-shot)    | **79.7**             | 63.3          |
| BBH (3-shot CoT)  | **76.3**             | 71.5          |
| MATH (0-shot CoT) | **64.7**             | 50.1          |
| GPQA (0-shot)     | **33.3**             | 29.3          |

- We use `ppl` for the MCQ evaluation on base model.
- The evaluation results were obtained from [OpenCompass](https://github.com/open-compass/opencompass) , and evaluation configuration can be found in the configuration files provided by [OpenCompass](https://github.com/open-compass/opencompass).
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/open-compass/opencompass), so please refer to the latest evaluation results of [OpenCompass](https://github.com/open-compass/opencompass).
- \* means the result is copied from the original paper.
