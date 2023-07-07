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
生成原始数据的`bin`和`meta`文件需要使用`tokenizer`，我们通过在`tools/tokenizer.py`中指定模型参数路径的方式来导入tokenizer模型。目前我们提供了`V7.model`来生成tokens。若想使用不同的模型，可直接修改`tokernizer.py`中的模型参数路径。

我们可以运行以下命令生成原始数据对应的`bin`和`meta`文件，其中参数`raw_data_name`表示原始数据集的文件名称，`input_file_type`表示原始数据集的文件格式，我们目前支持`txt`、`json`和`jsonl`这三种格式，`bin`表示生成的`bin`文件的保存路径。
```bash
$ python tools/tokenizer.py --raw_data_name your_raw_data_file_name(without suffix) --input_file_type 'text' or 'json' or 'jsonl' --bin your_output_bin_path
```

下面是一个数据处理的例子（这里只给出了`txt`格式的数据处理例子，`json`和`jsonl`的数据处理流程和`txt`的完全一致）：

给定一个包含原始数据集的文件`raw_data.txt`，原始数据集如下所示：
```bash
感恩生活中的每一个细节，才能真正体会到幸福的滋味。
梦想是人生的动力源泉，努力追逐，才能实现自己的目标。
学会宽容和理解，才能建立真正和谐的人际关系。
```

接下来，我们可以通过运行以下命令来生成`bin`和`meta`文件：
```bash
$ python tools/tokenizer.py --raw_data_name raw_data --input_file_type 'text' --bin cn/output.bin
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