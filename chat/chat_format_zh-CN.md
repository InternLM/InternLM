# 对话格式

[English](chat_format.md) | 简体中文

InternLM2-Chat 采用了全新的对话格式，以灵活地支持工具调用等更广泛的应用，并避免用户输入的攻击。新的对话格式和 [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) 格式类似，但是为了支持通用的智能体应用，在 `system`，`user`，`assistant` 的基础上，引入了 `environment` 角色。

## 基本结构

常规的对话结构一般包含 `system`，`user`，`assistant` 三个角色，采用如下格式进行多轮对话

```
<|im_start|>system
你是书生浦语2，一个无害的人工智能助手<|im_end|>
<|im_start|>user
你好呀<|im_end|>
<|im_start|>assistant
你好，我是书生浦语，请问有什么可以帮助你的吗<|im_end|>
```

其中 `<|im_start|>` 充当了每轮对话开始符，`<|im_end|>` 充当了当前轮对话结束符。每轮对话一般以 `<|im_start|>role` 开头，以模型输出的 `<|im_end|>` 结尾，role 代表 `system`，`user`，`assistant` 和 `environment` 角色。你可以参考[huggingface 上的代码](https://huggingface.co/internlm/internlm2_5-7b-chat/blob/main/modeling_internlm2.py#L1357)来了解对话历史的拼接。

目前，InternLM2-Chat 模型的词表中还维护了如下映射

- `<|im_start|>`：每个角色对话的开始符，在词表中对应的 token ID 是**92543**
- `<|im_end|>`：每个角色对话的结束符，在词表中对应的 token ID 是**92542**
- `<|action_start|>`：模型调用外部工具(代码解释器或外部插件)的开始符，在词表中对应的 token ID 是**92541**
- `<|action_end|>`：模型调用外部插件的结束符，在词表中对应的 token ID 是**92540**
- `<|interpreter|>`：代码解释器，在词表中对应的 token ID 是**92539**
- `<|plugin|>`：外部插件，常规的 tools，在词表中对应的 token ID 是**92538**

## 完整结构

InternLM2-Chat 的完整对话格式在上述基本结构的基础上还包含了针对通用智能体的设计，其核心目的是采用流式格式，使得同一套格式在支持各种类插件拓展和智能体环境的同时能够和通用对话兼容。

### 工具调用

InternLM2-Chat 的通用工具调用（function call）支持多种格式，尤其是**json 格式**以便捷下游应用。以此为例，一次完整的工具调用流程如下所示：

```
<|im_start|>system
你是书生浦语2，一个无害的人工智能助手<|im_end|>
<|im_start|>system name=<|plugin|>
[
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
]
<|im_end|>
<|im_start|>user
我想了解今天上海的天气<|im_end|>
<|im_start|>assistant
好的，我将为你查询上海的天气。<|action_start|><|plugin|>
{"name": "get_current_weather", "parameters": {"location": "Shanghai"}}<|action_end|><|im_end|>
<|im_start|>environment name=<|plugin|>
{"temperature": 22}<|im_end|>
<|im_start|>assistant
上海的天气是 22 摄氏度<|im_end|>
```

1. 首先，在系统提示中以 json 的格式提供工具调用的格式和字段描述。内容以 `<|im_start|>system name=<|plugin|>\n`开头，`<|im_end|>` 结尾，`name=<|plugin|>` 体现了这是来自工具的指令。InternLM2-Chat 支持 system 角色对模型的提示和约束多次出现。所以我们会看到前面还有关于对话的要求。
2. 模型在接受到用户指令后，会以流式的形式调用工具，及自然地生成文字进行思考/回应用户，然后输出`<|action_start|><|plugin|>`。`<|action_start|>`表示要调用外部插件，同时 `<|plugin|>` 表示调用的是常规工具。然后模型输出 json 格式的工具调用内容，再以 `<|action_end|>` 表示工具调用结束。
3. 系统返回工具调用结果，以 `<|im_start|>environment name=<|plugin|>`开头，表示是来自环境关于工具调用的反馈，内容是 json 格式，`{"temperature": 22}`，以`<|im_end|>`结尾。

### 代码解释器

InternLM2-Chat 支持使用代码解释器解决复杂数学问题和进行数据分析。整个调用流程和通用工具调用类似，如下所示

````
<|im_start|>system
你是书生浦语2，一个无害的人工智能助手<|im_end|>
<|im_start|>system name=<|interpreter|>
你现在可以使用一个支持 Python 代码执行的 Jupyter 笔记本环境。只需向 python 发送代码，即可在这个有状态环境中进行运行。这个功能适用于:
- 数据分析或处理（如数据操作和图形制作）
- 复杂计算（如数学和物理问题）
- 编程示例（用于理解编程概念或语言特性）
- 文本处理和分析（包括文本分析和自然语言处理）
- 机器学习和数据科学（模型训练和数据可视化展示）
- 文件操作和数据导入（处理CSV、JSON等格式文件）
<|im_start|>user
请帮我对该数据集进行数据处理并可视化。
<|im_end|>
<|im_start|>user name=file
[{"path": "data.csv", size='10K'}]<|im_end|>
<|im_start|>assistant
我已经帮您处理了数据并进行了可视化。

<|action_start|><|interpreter|>
```python
import plotly.express as px
import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv('data.csv')

# Create a scatter plot of rainfall vs wind direction
fig = px.scatter(df, x='WindDir9am', y='Rainfall', color='WindDir3pm',
                 labels={'WindDir9am': 'Wind Direction 9am', 'Rainfall': 'Rainfall', 'WindDir3pm': 'Wind Direction 3pm'},
                 title='Rainfall vs Wind Direction',
                 template='plotly_dark',
                 width=600, height=500)

# Add a hover effect to show the date
fig.update_traces(hovertemplate='<b>Date: %{text}</b><br>Wind Direction 9am: %{x}<br>Rainfall: %{y}<br>Wind Direction 3pm: %{marker.color}')

# Show the plot
fig.show()
```<|action_end|>
<|im_end|>
<|im_start|>environment name=<|interpreter|>
![image](xxx.png)
<|im_end|>
<|im_start|>assistant
由于代码输出没有包含在这里，我无法提供具体的图表内容。但是，如果代码执行无误，它会显示一个极坐标图，其中包含两个填充区域，一个代表上午9点的风向与降雨量的关系，另一个代表下午3点的风向与降雨量的关系。每个方向上的值是根据提供的数据集计算得到的平均降雨量。图表应该有一个清晰的标题、图例，并且能够直观地比较不同风向下的降雨量。由于使用了暗色主题，图表的整体外观应该是深色背景上的亮色线条和填充区域。<|im_end|>
<|im_start|>user
你能解释一下代码吗<|im_end|>
<|im_start|>assistant
我使用的是Python的Plotly库来创建一个极坐标图，以展示风向与降雨量之间的关系。代码中首先创建了一个Plotly的Figure对象。接着，我添加了两个`Scatterpolar`追踪。第一个追踪展示的是上午9点的风向与降雨量的平均值的关系，第二个追踪展示的是下午3点的风向与降雨量的平均值的关系。在这两个追踪中，`r`参数代表降雨量的平均值，`theta`参数代表风向。`fill='toself'`表示图形以自身为边界进行填充。然后我更新了图表的布局，隐藏了径向轴，添加了图例，设置了标题和标题字体，选择了暗色主题，并设定了图表的背景色、宽度和高度。最后，使用`fig.show()`展示了图表。
<|im_end|>
````

1. 首先在系统提示中提供代码解释器的格式和字段描述。内容以 `<|im_start|>system name=<|interpreter|>\n`开头，`<|im_end|>` 结尾，`name=<|interpreter|>` 体现了这是来自代码解释器的指令。InternLM2-Chat 支持 system 角色对模型的提示和约束多次出现。所以我们会看到前面还有关于对话的要求。
2. 用户可以上传一个文件，并对模型提出要求，文件的上传会以单独的形式向模型发出一条指令，以 `<|im_start|>user name=file` 开头，以 json 形式给出路径和文件大小`[{"path": "data.csv", size='10K'}]`，以 `<|im_end|>`结尾。
3. 模型在接受到用户指令后，会以流式的形式调用工具，及自然地生成文字进行思考/回应用户，然后输出`<|action_start|><|interpreter|>`。`<|action_start|>`表示要调用外部插件，同时 `<|interpreter|>` 表示调用的是代码解释器。然后模型输出 markdown 中 python 代码块格式代码内容，再以 `<|action_end|>` 表示工具调用结束。
4. 系统会执行代码块中的代码，然后返回调用结果，以 `<|im_start|>environment name=<|interpreter|>`开头，表示是来自环境关于代码解释器执行的输出，以`<|im_end|>`结尾。

### 同时使用工具和代码解释器

InternLM2-Chat 能够在一个对话过程中自主选择调用工具或代码解释器。在工具和代码解释器同时开启的情况下，只需要将各自的系统提示合并在一起给模型即可。一个调用工具和代码解释器的对话历史样例如下。

````
<|im_start|>system
你是书生浦语2，一个无害的人工智能助手<|im_end|>
<|im_start|>system name=<|interpreter|>
你现在可以使用一个支持 Python 代码执行的 Jupyter 笔记本环境。只需向 python 发送代码，即可在这个有状态环境中进行运行。这个功能适用于:
- 数据分析或处理（如数据操作和图形制作）
- 复杂计算（如数学和物理问题）
- 编程示例（用于理解编程概念或语言特性）
- 文本处理和分析（包括文本分析和自然语言处理）
- 机器学习和数据科学（模型训练和数据可视化展示）
- 文件操作和数据导入（处理CSV、JSON等格式文件）
<|im_start|>system name=<|plugin|>
[
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
]
<|im_end|>
<|im_start|>user
请帮我对该数据集进行数据处理并可视化。
<|im_end|>
<|im_start|>user name=file
[{"path": "data.csv", size='10K'}]<|im_end|>
<|im_start|>assistant
我已经帮您处理了数据并进行了可视化。

<|action_start|><|interpreter|>
```python
import plotly.express as px
import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv('data.csv')

# Create a scatter plot of rainfall vs wind direction
fig = px.scatter(df, x='WindDir9am', y='Rainfall', color='WindDir3pm',
                 labels={'WindDir9am': 'Wind Direction 9am', 'Rainfall': 'Rainfall', 'WindDir3pm': 'Wind Direction 3pm'},
                 title='Rainfall vs Wind Direction',
                 template='plotly_dark',
                 width=600, height=500)

# Add a hover effect to show the date
fig.update_traces(hovertemplate='<b>Date: %{text}</b><br>Wind Direction 9am: %{x}<br>Rainfall: %{y}<br>Wind Direction 3pm: %{marker.color}')

# Show the plot
fig.show()
```<|action_end|>
<|im_end|>
<|im_start|>environment name=<|interpreter|>
![image](xxx.png)
<|im_end|>
<|im_start|>assistant
由于代码输出没有包含在这里，我无法提供具体的图表内容。但是，如果代码执行无误，它会显示一个极坐标图，其中包含两个填充区域，一个代表上午9点的风向与降雨量的关系，另一个代表下午3点的风向与降雨量的关系。每个方向上的值是根据提供的数据集计算得到的平均降雨量。图表应该有一个清晰的标题、图例，并且能够直观地比较不同风向下的降雨量。由于使用了暗色主题，图表的整体外观应该是深色背景上的亮色线条和填充区域。<|im_end|>
<|im_start|>user
你能解释一下代码吗<|im_end|>
<|im_start|>assistant
我使用的是Python的Plotly库来创建一个极坐标图，以展示风向与降雨量之间的关系。代码中首先创建了一个Plotly的Figure对象。接着，我添加了两个`Scatterpolar`追踪。第一个追踪展示的是上午9点的风向与降雨量的平均值的关系，第二个追踪展示的是下午3点的风向与降雨量的平均值的关系。在这两个追踪中，`r`参数代表降雨量的平均值，`theta`参数代表风向。`fill='toself'`表示图形以自身为边界进行填充。然后我更新了图表的布局，隐藏了径向轴，添加了图例，设置了标题和标题字体，选择了暗色主题，并设定了图表的背景色、宽度和高度。最后，使用`fig.show()`展示了图表。
<|im_end|>
<|im_start|>user
我想了解今天上海的天气<|im_end|>
<|im_start|>assistant
好的，我将为你查询上海的天气。<|action_start|><|plugin|>
{"name": "get_current_weather", "parameters": {"location": "Shanghai"}}<|action_end|><|im_end|>
<|im_start|>environment name=<|plugin|>
{"temperature": 22}<|im_end|>
<|im_start|>assistant
上海的天气是 22 摄氏度<|im_end|>
````
