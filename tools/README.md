本目录提供辅助模型训练的一些工具，文件结构如下所示：

```bash
├── transformers  # 适配hugging face的transformers的一些工具
│   ├── configuration_internlm.py  # config适配工具
│   ├── modeling_internlm.py  # model适配工具
│   ├── tokenization_internlm.py  # tokenizer适配工具
│   └── convert2hf.py  # 模型适配hugging face工具
└── tokenizer.py  # 将原始数据转换成bin和meta文件的工具
```

# tokenizer.py

生成原始数据的`bin`和`meta`文件需要使用`tokenizer`，我们通过在`tools/tokenizer.py`中指定模型参数路径的方式来导入tokenizer模型。目前我们提供了`V7_sft.model`来生成tokens。若想使用不同的模型，可直接修改`tokernizer.py`中的模型参数路径。

可以运行以下命令生成原始数据对应的`bin`和`meta`文件，其中参数`text_input_path`表示原始文本数据路径，目前支持`txt`、`json`和`jsonl`三种输入格式，`bin_output_path`表示生成的`bin`文件的保存路径。

```bash
$ python tools/tokenizer.py --text_input_path your_input_text_path --bin_output_path your_output_bin_path
```

下面是一个数据处理的例子：

给定一个包含原始数据集的文件`raw_data.txt`，原始数据集如下所示：

```bash
感恩生活中的每一个细节，才能真正体会到幸福的滋味。
梦想是人生的动力源泉，努力追逐，才能实现自己的目标。
学会宽容和理解，才能建立真正和谐的人际关系。
```

可以通过运行以下命令来生成`bin`和`meta`文件：
```bash
$ python tools/tokenizer.py --text_input_path raw_data.txt --bin_output_path cn/output.bin
```

需要注意的是，生成的`bin`文件需要保存在`cn`或者`en`或者`code`或者`ja`或者`ar`或者`kaoshi`这五个目录下，以区分数据集的类型。

其中，`cn`表示中文数据集；`en`表示英文数据集；`code`表示代码数据集；`ja`表示日语数据集；`ar`表示阿拉伯语数据集；`kaoshi`表示考试数据集。

生成的bin文件的格式如下：

```python
{"tokens": [73075, 75302, 69522, 69022, 98899, 67713, 68015, 81269, 74637, 75445, 99157]}
{"tokens": [69469, 60355, 73026, 68524, 60846, 61844, 98899, 67775, 79241, 98899, 67713, 67800, 67453, 67838, 99157]}
{"tokens": [68057, 79017, 60378, 68014, 98899, 67713, 67990, 68015, 70381, 67428, 61003, 67622, 99157]}
```

`bin`文件中的每一行均对应原始数据集中的每一个句子，表示每个句子的`token`（下文将用sequence指定）。

生成的`meta`文件的格式如下：

```bash
(0, 11), (90, 15), (208, 13)
```

在`meta`文件中，每个元组对应着`bin`文件中每一个`sequence`的元信息。其中，元组的第一个元素表示每个`sequence`在所有`sequence`中的`starting index`，第二个元素表示每个`sequence`中有多少个`tokens`。

例如，对于第一个`sequence`，`starting index`为 0，有 11 个`tokens`；对于第二个`sequence`，由于第一个`sequence`转换为`string`后的长度为`89`，因此它的`starting index`为 90，有 15 个`tokens`。

`json`和`jsonl`类型的文件的`bin`和`meta`文件格式和`txt`一致，此处不再赘叙。

# pal_inference.py

在 [GSM8K](https://huggingface.co/datasets/gsm8k) 数据集上使用 [PAL](https://github.com/reasoning-machines/pal) 范式推理，使模型编写代码并通过 Python 解释器执行来解决数学问题。其用法如下：

```python
# 用法:
python pal_inference.py <model> <out_dir> [--dataset <dataset>] [--max_length <length>] [--top_p <threshold>] [--eoh <end token>] [--eoa <end token>] [--eos <end token>] [--temperature <temp>] [--time_out <time>] [--verbose, -v] [--append, -a]

# 参数:
# <model>                   用于推理的模型的路径。
# <out_dir>                 生成代码将保存在指定的输出文件夹中。

# 可选参数:
# --dataset <dataset>       用于代码生成的数据集名称（默认：gsm8k）。
# --max_length <length>     模型最大输入 token 长度（默认：2048）。
# --top_p <threshold>       候选 token 相加的概率阈值（默认：0.8）。
# --eoh <end token>         用户输入结束标识符 (默认: "") 。
# --eoa <end token>         模型输入结束标识符 (默认: "") 。
# --eos <end token>         系统输入结束标识符. (默认: "") 。
# --temperature， -t <temp> 生成过程中的采样温度（默认：1.0）。
# --time_out <time>         执行生成的代码的最大时间（秒）（默认：100）。
# --verbose, -v             打印代码错误信息（可选）。
# --append, -a              将输出追加到历史结果中（可选）。
```

以下是使用示例：

```bash
python tools/pal_inference.py internlm/internlm-chat-7k ./output -v
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

InternLM 在 GSM8K 数据集中带工具和不带工具的性能表现：

| Method   | **InternLM-Chat-7B** |
| -------- | -------------------- |
| w/o tool | 34.5                 |
| w tool   | 39.2                 |
