# InternLM-Chat Agent

English | [简体中文](README_zh-CN.md)

## Introduction

InternLM2.5-Chat, open sourced on June 30, 2024, further enhances its capabilities in code interpreter and general tool utilization. With improved and more generalized instruction understanding, tool selection, and reflection abilities, InternLM2.5-Chat can more reliably support complex agents and multi-step tool calling for more intricate tasks. When combined with a code interpreter, InternLM2.5-Chat obtains comparable results to GPT-4 on MATH. Leveraging strong foundational capabilities in mathematics and tools, InternLM2.5-Chat provides practical data analysis capabilities.

The results of InternLM2.5-Chat on math code interpreter is as below:

|       Models        | Tool-Integrated | MATH |
| :-----------------: | :-------------: | :--: |
|  InternLM2-Chat-7B  |       w/        | 45.1 |
| InternLM2-Chat-20B  |       w/        | 51.2 |
| InternLM2.5-7B-Chat |       w/        | 63.0 |
| gpt-4-0125-preview  |       w/o       | 64.2 |

## Usages

We offer an example using [Lagent](lagent.md) to build agents based on InternLM2.5-Chat to call the code interpreter. Firstly install the extra dependencies:

```bash
pip install -r requirements.txt
```

Run the following script to perform inference and evaluation on MATH test.

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

`output_path` is a jsonl format file to save the inference results. Each line is like

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

Once it is prepared, just skip the inference stage as follows.

```bash
python streaming_inference.py \
  --output_path=math_lmdeploy.jsonl \
  --no-do_infer \
  --do_eval
```

Please refer to [`streaming_inference.py`](streaming_inference.py) for more information about the arguments.
