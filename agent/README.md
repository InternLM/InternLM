# InternLM-Chat Agent

English | [简体中文](README_zh-CN.md)

## Introduction

On August 22, 2023, the Shanghai Artificial Intelligence Laboratory open-sourced InternLM-Chat-7B v1.1, the first open-dialogue model with code interpretation capabilities. It supports external tools such as the Python interpreter and can fetch real-time information through search engines.

InternLM2-Chat, open-sourced on January 17, 2024, further enhances its capabilities in code interpretation and general tool invocation. With improved and more generalized instruction understanding, tool selection, and result reflection, the new model can more reliably support the construction of complex intelligent agents. It facilitates effective multi-round invocation of tools and accomplishes more intricate tasks. The model exhibits decent computational and reasoning abilities even without external tools, surpassing ChatGPT in mathematical performance. When combined with a code interpreter, InternLM2-Chat-20B achieves a level comparable to GPT-4 on GSM8K and MATH. Leveraging strong foundational capabilities in mathematics and tools, InternLM2-Chat provides practical data analysis capabilities.

|       | GSM8K | MATH |
| :---: | :---: | :--: |
| InternLM2-Chat-20B | 79.6 | 32.5 |
| InternLM2-Chat-20B with Code Interpreter  | 84.5 | 51.2 |
| ChatGPT (GPT-3.5) | 78.2 | 28.0 |
| GPT-4 | 91.4 | 45.8 |

## Experience

We offer examples using [Lagent](lagent.md) to build intelligent agents based on InternLM2-Chat, calling code interpreters or searching tools. Additionally, we provide a sample using [PAL to evaluate GSM8K math problems](pal_inference.md) with InternLM-Chat-7B.
