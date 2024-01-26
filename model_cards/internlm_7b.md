# InternLM-7B Model Card

## Introduction

InternLM-7B contains a 7 billion parameter base model and a chat model tailored for practical scenarios. The model has the following characteristics:

- It leverages trillions of high-quality tokens for training to establish a powerful knowledge base.
- It supports an 8k context window length, enabling longer input sequences and stronger reasoning capabilities.
- It provides a versatile toolset for users to flexibly build their own workflows.

## Model Zoo

| Model                | Transformers(HF)                            | ModelScope(HF)                            | OpenXLab(HF)                            | OpenXLab(Original)                            | Release Date |
| -------------------- | ------------------------------------------- | ----------------------------------------- | --------------------------------------- | --------------------------------------------- | ------------ |
| **InternLM Chat 7B** | [🤗internlm/internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) | [<img src="../assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-original) | 2023-12-12   |
| **InternLM 7B**      | [🤗internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) | [<img src="../assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-7b-original) | 2023-07-06   |

## Performance Evaluation

We conducted a comprehensive evaluation of InternLM using the open-source evaluation tool [OpenCompass](https://github.com/internLM/OpenCompass/). The evaluation covered five dimensions of capabilities: disciplinary competence, language competence, knowledge competence, inference competence, and comprehension competence. Here are some of the evaluation results, and you can visit the [OpenCompass leaderboard](https://opencompass.org.cn/rank) for more evaluation results.

| Datasets\\Models | **InternLM-Chat-7B** | **InternLM-7B** | LLaMA-7B | Baichuan-7B | ChatGLM2-6B | Alpaca-7B | Vicuna-7B |
| ---------------- | -------------------- | --------------- | -------- | ----------- | ----------- | --------- | --------- |
| C-Eval(Val)      | 52.0                 | 53.4            | 24.2     | 42.7        | 50.9        | 28.9      | 31.2      |
| MMLU             | 52.6                 | 51.0            | 35.2\*   | 41.5        | 46.0        | 39.7      | 47.3      |
| AGIEval          | 46.4                 | 37.6            | 20.8     | 24.6        | 39.0        | 24.1      | 26.4      |
| CommonSenseQA    | 80.8                 | 59.5            | 65.0     | 58.8        | 60.0        | 68.7      | 66.7      |
| BUSTM            | 80.6                 | 50.6            | 48.5     | 51.3        | 55.0        | 48.8      | 62.5      |
| CLUEWSC          | 81.8                 | 59.1            | 50.3     | 52.8        | 59.8        | 50.3      | 52.2      |
| MATH             | 5.0                  | 7.1             | 2.8      | 3.0         | 6.6         | 2.2       | 2.8       |
| GSM8K            | 36.2                 | 31.2            | 10.1     | 9.7         | 29.2        | 6.0       | 15.3      |
| HumanEval        | 15.9                 | 10.4            | 14.0     | 9.2         | 9.2         | 9.2       | 11.0      |
| RACE(High)       | 80.3                 | 57.4            | 46.9\*   | 28.1        | 66.3        | 40.7      | 54.0      |

- The evaluation results were obtained from [OpenCompass 20230706](https://github.com/internLM/OpenCompass/) (some data marked with \*, which means come from the original papers), and evaluation configuration can be found in the configuration files provided by [OpenCompass](https://github.com/internLM/OpenCompass/).
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/internLM/OpenCompass/), so please refer to the latest evaluation results of [OpenCompass](https://github.com/internLM/OpenCompass/).
