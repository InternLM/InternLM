# InternLM-Chat Agent

English | [简体中文](README_zh-CN.md)

## Introduction

InternLM-Chat-7B v1.1 has been released as the first open-source model with code interpreter capabilities, supporting external tools such as Python code interpreter and search engine.

InternLM2-Chat, open sourced on January 17, 2024, further enhances its capabilities in code interpreter and general tool utilization. With improved and more generalized instruction understanding, tool selection, and reflection abilities, InternLM2-Chat can more reliably support complex agents and multi-step tool calling for more intricate tasks. InternLM2-Chat exhibits decent computational and reasoning abilities even without external tools, surpassing ChatGPT in mathematical performance. When combined with a code interpreter, InternLM2-Chat-20B obtains comparable results to GPT-4 on GSM8K and MATH. Leveraging strong foundational capabilities in mathematics and tools, InternLM2-Chat provides practical data analysis capabilities.

The results of InternLM2-Chat-20B on math code interpreter is as below:

|                                          | GSM8K | MATH |
| :--------------------------------------: | :---: | :--: |
|            InternLM2-Chat-20B            | 79.6  | 32.5 |
| InternLM2-Chat-20B with Code Interpreter | 84.5  | 51.2 |
|            ChatGPT (GPT-3.5)             | 78.2  | 28.0 |
|                  GPT-4                   | 91.4  | 45.8 |

## Usages

We offer examples using [Lagent](lagent.md) to build agents based on InternLM2-Chat to call code interpreter or search API. Additionally, we provide an example code using [PAL to evaluate GSM8K math problems](pal_inference.md) with InternLM-Chat-7B.
