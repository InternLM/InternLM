# Lagent 智能体流式推理

[English](streaming_inference.md) | 简体中文

推荐使用 [Lagent](https://github.com/InternLM/lagent) 框架，该工具包实现了多种类型智能体并集成了常见工具，包括 Python 代码解释器。

我们基于此提供一个在 GSM8K 和 MATH 测试集上推理和评估的脚本，使用方式如下：

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
