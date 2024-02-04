# InternLM2-20B Model Card

## Introduction

InternLM2-1.8B is the 1.8 billion parameter version of the second generation InternLM series. In order to facilitate user use and research, InternLM2-1.8B has three versions of open-source models. They are:

- InternLM2-1.8B: Foundation models with high quality and high adaptation flexibility, which serve as a good starting point for downstream deep adaptations.
- InternLM2-Chat-1.8B-SFT: Chat model after supervised fine-tuning (SFT) on InternLM2-1.8B. 
- InternLM2-Chat-1.8B: Further aligned on top of InternLM2-Chat-1.8B-SFT through online RLHF. InternLM2-Chat-1.8B exhibits better instruction following, chat experience, and function calling, which is recommended for downstream applications.

The base model of InternLM2 has the following technical features:

- Effective support for ultra-long contexts of up to 200,000 characters: The model nearly perfectly achieves "finding a needle in a haystack" in long inputs of 200,000 characters. It also leads among open-source models in performance on long-text tasks such as LongBench and L-Eval.
- Comprehensive performance enhancement: Compared to the previous generation model, it shows significant improvements in various capabilities, including reasoning, mathematics, and coding.

## Model Zoo

| Model | Transformers(HF) | Release Date |
|---------------------------|------------------------------------------------------------------------------------------|--------------|
| **InternLM2 1.8B** | [🤗internlm/internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b) | 2024-01-31 |
| **InternLM2 Chat 1.8B SFT**     | [🤗internlm/internlm2-chat-1_8b-sft](https://huggingface.co/internlm/internlm2-chat-1_8b-sft)         | 2024-01-31   |

## Performance Evaluation

We have evaluated InternLM2 on several important benchmarks using the open-source evaluation tool [OpenCompass](https://github.com/open-compass/opencompass). Some of the evaluation results are shown in the table below. You are welcome to visit the [OpenCompass Leaderboard](https://opencompass.org.cn/rank) for more evaluation results.

| Dataset\Models | InternLM2-1.8B | InternLM2-Chat-1.8B-SFT | InternLM2-7B | InternLM2-Chat-7B |
| :---: | :---: | :---: | :---: | :---: |
| MMLU | 46.9 | 47.1 | 65.8 | 63.7 |
| AGIEval | 33.4 | 38.8 | 49.9 | 47.2 |
| BBH | 37.5 | 35.2 | 65.0 | 61.2 |
| GSM8K | 31.2 | 39.7 | 70.8 | 70.7 |
| MATH | 5.6 | 11.8 | 20.2 | 23.0 |
| HumanEval | 25.0 | 32.9 | 43.3 | 59.8 |
| MBPP(Sanitized) | 22.2 | 23.2 | 51.8 | 51.4 |


- The evaluation results were obtained from [OpenCompass](https://github.com/open-compass/opencompass) , and evaluation configuration can be found in the configuration files provided by [OpenCompass](https://github.com/open-compass/opencompass). 
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/open-compass/opencompass), so please refer to the latest evaluation results of [OpenCompass](https://github.com/open-compass/opencompass).
