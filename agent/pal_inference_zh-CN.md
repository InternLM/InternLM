# 采用 PAL 在 InternLM1-Chat 上评测 GSM8K

[English](pal_inference.md) | 简体中文

在 [GSM8K](https://huggingface.co/datasets/gsm8k) 数据集上使用 [PAL](https://github.com/reasoning-machines/pal) 范式推理，使模型编写代码并通过 Python 解释器执行来解决数学问题。其用法如下：

```bash
python pal_inference.py \
    <model> \
    <out_dir> \
    [--dataset <dataset>] \
    [--max_length <length>] \
    [--top_p <threshold>] \
    [--eoh <end token>] \
    [--eoa <end token>] \
    [--eos <end token>] \
    [--temperature <temp>] \
    [--time_out <time>] \
    [--verbose, -v] \
    [--append, -a]
```

参数说明:

|           参数            |                    说明                     |
| :-----------------------: | :-----------------------------------------: |
|         \<model>          |            用于推理的模型的路径             |
|        \<out_dir>         |     生成代码将保存在指定的输出文件夹中      |
|    --dataset <dataset>    |   用于代码生成的数据集名称（默认：gsm8k）   |
|   --max_length <length>   |    模型最大输入 token 长度（默认：2048）    |
|    --top_p <threshold>    |   候选 token 相加的概率阈值（默认：0.8）    |
|     --eoh <end token>     |        用户输入结束标识符 (默认: "")        |
|     --eoa <end token>     |        模型输入结束标识符 (默认: "")        |
|     --eos <end token>     |       系统输入结束标识符. (默认: "")        |
| --temperature， -t <temp> |      生成过程中的采样温度（默认：1.0）      |
|     --time_out <time>     | 执行生成的代码的最大时间（秒）（默认：100） |
|       --verbose, -v       |          打印代码错误信息（可选）           |
|       --append, -a        |       将输出追加到历史结果中（可选）        |

简单的使用示例如下：

```bash
python tools/pal_inference.py internlm/internlm-chat-7b ./output -v
```

其输出文件每一行包括输入的问题，正确答案，执行答案，得分，以及模型生成的 Python 代码块：

````json
{
    "question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "target": 18.0,
    "answer": 18.0,
    "score": 1,
    "generation": ["```python\ndef solution():\n    eggs_per_day = 16\n    eggs_per_breakfast = 3\n    eggs_per_muffin = 4\n    eggs_used = eggs_per_day - eggs_per_breakfast - eggs_per_muffin\n    eggs_sold = eggs_used\n    price_per_egg = 2\n    eggs_made = eggs_sold * price_per_egg\n    result = eggs_made\n    return result\n```"]
}
````

InternLM 在 GSM8K 数据集中带工具和不带工具的性能表现如下表所示。

| Method   | **InternLM-Chat-7B** |
| -------- | -------------------- |
| w/o tool | 34.5                 |
| w tool   | 39.2                 |
