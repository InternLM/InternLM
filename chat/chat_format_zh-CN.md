# 对话格式

[English](chat_format.md) | 简体中文

InternLM2-Chat 采用了全新的对话格式，以灵活地支持工具调用等更广泛的应用，并避免用户输入的攻击。新的对话格式和 [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) 格式类似，但是为了支持通用的智能体应用，在 `system`，`user`，`assistant` 的基础上，引入了 `environment` 角色。

## 基本结构

常规的对话结构一般包含 `system`，`user`，`assistant` 三个角色，采用如下格式进行多轮对话

```
[UNUSED_TOKEN_146]system
你是书生浦语2，一个无害的人工智能助手[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]user
你好呀[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
你好，我是书生浦语，请问有什么可以帮助你的吗[UNUSED_TOKEN_145]
```

其中 `[UNUSED_TOKEN_146]` 充当了每轮对话开始符，`[UNUSED_TOKEN_145]` 充当了当前轮对话结束符。每轮对话一般以 `[UNUSED_TOKEN_146]role` 开头，以模型输出的 `[UNUSED_TOKEN_145]` 结尾，role 代表 `system`，`user`，`assistant` 和 `environment` 角色。目前，InternLM2-Chat 模型的词表中还维护了如下映射

- `[UNUSED_TOKEN_146]`：每个角色对话的开始符
- `[UNUSED_TOKEN_145]`：每个角色对话的结束符
- `[UNUSED_TOKEN_144]`：模型调用外部插件的开始符
- `[UNUSED_TOKEN_143]`：模型调用外部插件的结束符
- `[UNUSED_TOKEN_142]`：代码解释器
- `[UNUSED_TOKEN_141]`：外部插件，常规的 tools

## 完整结构

InternLM2-Chat 的完整对话格式在上述基本结构的基础上还包含了针对通用智能体的设计，其核心目的是采用流式格式，使得同一套格式在支持各种类插件拓展和智能体环境的同时能够和通用对话兼容。通用的智能体对话状态如下所示

```
[UNUSED_TOKEN_146]system
你是书生浦语2，一个无害的人工智能助手[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]system name=[UNUSED_TOKEN_141]
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
[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]user
请帮我对该数据集进行数据处理并可视化。
[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]user name=file
{"path": "data.csv"}[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
我已经帮您处理了数据并进行了可视化。

[UNUSED_TOKEN_144][UNUSED_TOKEN_142]
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
```[UNUSED_TOKEN_143]
[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]environment name=[UNUSED_TOKEN_142]
[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
由于代码输出没有包含在这里，我无法提供具体的图表内容。但是，如果代码执行无误，它会显示一个极坐标图，其中包含两个填充区域，一个代表上午9点的风向与降雨量的关系，另一个代表下午3点的风向与降雨量的关系。每个方向上的值是根据提供的数据集计算得到的平均降雨量。图表应该有一个清晰的标题、图例，并且能够直观地比较不同风向下的降雨量。由于使用了暗色主题，图表的整体外观应该是深色背景上的亮色线条和填充区域。[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]user
你能解释一下代码吗[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
我使用的是Python的Plotly库来创建一个极坐标图，以展示风向与降雨量之间的关系。代码中首先创建了一个Plotly的Figure对象。接着，我添加了两个`Scatterpolar`追踪。第一个追踪展示的是上午9点的风向与降雨量的平均值的关系，第二个追踪展示的是下午3点的风向与降雨量的平均值的关系。在这两个追踪中，`r`参数代表降雨量的平均值，`theta`参数代表风向。`fill='toself'`表示图形以自身为边界进行填充。然后我更新了图表的布局，隐藏了径向轴，添加了图例，设置了标题和标题字体，选择了暗色主题，并设定了图表的背景色、宽度和高度。最后，使用`fig.show()`展示了图表。
[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]user
我想了解今天上海的天气[UNUSED_TOKEN_145]
[UNUSED_TOKEN_144][UNUSED_TOKEN_141]
{"name": "get_current_weather", "parameters": {"location": "上海"}}[UNUSED_TOKEN_143]
[UNUSED_TOKEN_145]
```
