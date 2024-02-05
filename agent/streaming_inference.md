# Inference with Streaming Agents in Lagent

English | [简体中文](streaming_inference_zh-CN.md)

[Lagent](https://github.com/InternLM/lagent) is strongly recommended for agent construction. It supports multiple types of agents and is integrated with commonly used tools, including code interpreters.

We provide a script for inference and evaluation on GSM8K and MATH test. The usage is as follows:

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

`output_path` is a jsonl format file to save the inference results. Each line is like:

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

Once it is prepared, just skip the inference stage as follows:

```bash
python streaming_inference.py \
  --output_path=math_lmdeploy.jsonl \
  --no-do_infer \
  --do_eval
```

Please refer to [`streaming_inference.py`](streaming_inference.py) for more information about the arguments.
