# InternLM-Chat 智能体

[English](README.md) | 简体中文

## 简介

InternLM-Chat-7B v1.1 是首个具有代码解释能力的开源对话模型，支持 Python 解释器和搜索引擎等外部工具。

InternLM2-Chat 进一步提高了它在代码解释和通用工具调用方面的能力。基于更强和更具有泛化性的指令理解、工具筛选与结果反思等能力，新版模型可以更可靠地支持复杂智能体的搭建，支持对工具进行有效的多轮调用，完成较复杂的任务。模型在不使用外部工具的条件下已具备不错的计算能力和推理能力，数理表现超过 ChatGPT；在配合代码解释器（code-interpreter）的条件下，InternLM2-Chat-20B 在 GSM8K 和 MATH 上可以达到和 GPT-4 相仿的水平。基于在数理和工具方面强大的基础能力，InternLM2-Chat 提供了实用的数据分析能力。

以下是 InternLM2-Chat-20B 在数学代码解释器上的结果。

|                                     | GSM8K | MATH  |
| :---------------------------------: | :---: | :---: |
| InternLM2-Chat-20B 单纯依靠内在能力 | 79.6  | 32.5  |
|  InternLM2-Chat-20B 配合代码解释器  | 84.5  | 51.2  |
|          ChatGPT (GPT-3.5)          | 78.2  | 28.0  |
|                GPT-4                | 91.4  | 45.8  |

## 体验

我们提供了使用 [Lagent](lagent_zh-CN.md) 来基于 InternLM2-Chat 构建智能体调用代码解释器的例子。首先安装额外依赖：

```bash
pip install -r requirements.txt
```

运行以下脚本在 GSM8K 和 MATH 测试集上进行推理和评估：

```bash
python streaming_inference.py \
  --backend=lmdeploy \  # For HuggingFace models: hf
  --model_path=internlm/internlm2-chat-20b \
  --tp=2 \
  --temperature=0.0 \
  --dataset=math \
  --output_path=math_lmdeploy.jsonl \
  --do_eval
```

`output_path` 是一个存储推理结果的 jsonl 格式文件，每行形如：

```json
{
    "idx": 41, 
    "query": "The point $(a, b)$ lies on the line with the equation $3x + 2y = 12.$ When $a = 4$, what is the value of $b$?",
    "gt": "0",
    "pred": ["0"],
    "steps": [
        {
            "role": "language",
            "content": ""
        },
        {
            "role": "tool",
            "content": {
                "name": "IPythonInteractive",
                "parameters": {
                    "command": "```python\nfrom sympy import symbols, solve\n\ndef find_b():\n    x, y = symbols('x y')\n    equation = 3*x + 2*y - 12\n    b = solve(equation.subs(x, 4), y)[0]\n\n    return b\n\nresult = find_b()\nprint(result)\n```"
                }
            },
            "name": "interpreter"
        },
        {
            "role": "environment",
            "content": "0",
            "name": "interpreter"
        },
        {
            "role": "language",
            "content": "The value of $b$ when $a = 4$ is $\\boxed{0}$."
        }
    ],
    "error": null
}
```

如果已经准备好了该文件，可直接跳过推理阶段进行评估：

```bash
python streaming_inference.py \
  --output_path=math_lmdeploy.jsonl \
  --no-do_infer \
  --do_eval
```

请参考 [`streaming_inference.py`](streaming_inference.py) 获取更多关于参数的信息。
