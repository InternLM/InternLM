# InternLM-Chat 智能体

[English](README.md) | 简体中文

## 简介

在 2023 年 8 月 22 日，上海人工智能实验室开源了 InternLM-Chat-7B v1.1，是首个具有代码解释能力的开源对话模型，支持 Python 解释器等外部工具，并且可以通过搜索引擎获取实时信息。

2024 年 1 月 17 日开源的 InternLM2-Chat 进一步提高了在代码解释和通用工具调用方面的能力。基于更强和更具有泛化性的指令理解、工具筛选与结果反思等能力，新版模型可以更可靠地支持复杂智能体的搭建，支持对工具进行有效的多轮调用，完成较复杂的任务。模型在不使用外部工具的条件下已具备不错的计算能力和推理能力，数理表现超过 ChatGPT；在配合代码解释器（code-interpreter）的条件下，InternLM2-Chat-20B 在 GSM8K 和 MATH 上可以达到和 GPT-4 相仿的水平。基于在数理和工具方面强大的基础能力，InternLM2-Chat 提供了实用的数据分析能力。

## 体验

我们提供了使用 [Lagent](lagent_zh_cn.md) 来基于 InternLM2-Chat 构建智能体调用代码解释器或者搜索等工具的例子。同时，我们也提供了采用 [PAL 评测 GSM8K 数学题](pal_inference_zh-CN.md) InternLM-Chat-7B 的样例。
