# InternLM-Chat 智能体

[English](README.md) | 简体中文

## 简介

InternLM2.5-Chat 在代码解释和通用工具调用方面的能力得到进一步提升。基于更强和更具有泛化性的指令理解、工具筛选与结果反思等能力，新版模型可以更可靠地支持复杂智能体的搭建，支持对工具进行有效的多轮调用，完成较复杂的任务。在配合代码解释器（code-interpreter）的条件下，InternLM2.5-Chat 在 MATH 上可以达到和 GPT-4 相仿的水平。基于在数理和工具方面强大的基础能力，InternLM2.5-Chat 提供了实用的数据分析能力。

以下是 InternLM2.5-Chat 在数学代码解释器上的结果。

|        模型         | 是否集成工具 | MATH |
| :-----------------: | :----------: | :--: |
|  InternLM2-Chat-7B  |      w/      | 45.1 |
| InternLM2-Chat-20B  |      w/      | 51.2 |
| InternLM2.5-7B-Chat |      w/      | 63.0 |
| gpt-4-0125-preview  |     w/o      | 64.2 |

## 体验

我们提供了使用 [Lagent](lagent_zh-CN.md) 来基于 InternLM2.5-Chat 构建智能体调用代码解释器的例子。首先安装额外依赖：

```bash
pip install -r requirements.txt
```

运行以下脚本在 MATH 测试集上进行推理和评估：

```bash
python streaming_inference.py \
  --backend=lmdeploy \  # For HuggingFace models: hf
  --model_path=internlm/internlm2_5-7b-chat \
  --tp=1 \
  --temperature=1.0 \
  --top_k=1 \
  --output_path=math_lmdeploy.jsonl \
  --do_eval
```

`output_path` 是一个存储推理结果的 jsonl 格式文件，每行形如：

````json
{
    "problem": "The midpoint of the line segment between $(x,y)$ and $(2,4)$ is $(-7,0)$. Find $(x,y)$.",
    "level": "Level 3",
    "type": "Algebra",
    "solution": "Applying the midpoint formula gives  $$\\left(\\frac{2+x}{2},\\frac{4+y}{2}\\right)=(-7,0).$$Solving $\\frac{2+x}{2}=-7$ for $x$ and $\\frac{4+y}{2} = 0$ for $y$ we find $(x,y)$ to be $\\boxed{(-16,-4)}$.",
    "idx": 306,
    "query": "The midpoint of the line segment between $(x,y)$ and $(2,4)$ is $(-7,0)$. Find $(x,y)$.",
    "gt": "(-16,-4)",
    "pred": [
        "(-16,-4)"
    ],
    "steps": [
        {
            "role": "language",
            "content": "To find the coordinates of the point $(x,y)$, we can use the midpoint formula. The midpoint $M$ of a line segment with endpoints $(x_1, y_1)$ and $(x_2, y_2)$ is given by:\n\n\\[M = \\left(\\frac{x_1 + x_2}{2}, \\frac{y_1 + y_2}{2}\\right)\\]\n\nGiven that the midpoint is $(-7,0)$ and one endpoint is $(2,4)$, we can set up equations to solve for $x$ and $y$.\n\nLet's start by setting up the equations based on the midpoint formula:\n\n\\[\\frac{x + 2}{2} = -7\\]\n\\[\\frac{y + 4}{2} = 0\\]\n\nNow, let's solve these equations using Python to find the values of $x$ and $y$."
        },
        {
            "role": "tool",
            "content": {
                "name": "AsyncIPythonInteractiveManager",
                "parameters": {
                    "command": "```python\nfrom sympy import symbols, Eq, solve\n\n# Define symbols\nx, y = symbols('x y')\n\n# Define equations based on the midpoint formula\neq1 = Eq((x + 2)/2, -7)\neq2 = Eq((y + 4)/2, 0)\n\n# Solve equations\nsolution = solve((eq1, eq2), (x, y))\n\nsolution\n```"
                }
            },
            "name": "interpreter"
        },
        {
            "role": "environment",
            "content": "{x: -16, y: -4}",
            "name": "interpreter"
        },
        {
            "role": "language",
            "content": "After solving the equations, we find that the coordinates of the point $(x,y)$ are $(-16, -4)$. Therefore, the solution to the problem is:\n\n\\[\\boxed{(-16, -4)}\\]"
        }
    ],
    "error": null
}
````

如果已经准备好了该文件，可直接跳过推理阶段进行评估：

```bash
python streaming_inference.py \
  --output_path=math_lmdeploy.jsonl \
  --no-do_infer \
  --do_eval
```

请参考 [`streaming_inference.py`](streaming_inference.py) 获取更多关于参数的信息。
