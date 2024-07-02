# Chat Format

English | [简体中文](chat_format_zh-CN.md)

InternLM2-Chat adopts a new chat format to flexibly support a wider range of applications, such as tool invocation, while avoiding user input attacks. This new format is similar to the [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) format, but with an added `environment` role to support general-purpose AI applications, in addition to `system`, `user`, and `assistant`.

## Basic Structure

The regular chat structure usually contains three roles: `system`, `user`, and `assistant`, formatted as follows for multi-turn dialogues:

```
<|im_start|>system
You are InternLM2-Chat, a harmless AI assistant<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hello, I am InternLM2-Chat, how can I assist you?<|im_end|>
```

Here, `<|im_start|>` acts as the start token for each turn of dialogue, and `<|im_end|>` as the end token. Each turn of dialogue typically starts with `<|im_start|>role` and ends with the model's output `<|im_end|>`, where role represents `system`, `user`, `assistant`, and `environment`. You may refer to the [code in huggingface](https://huggingface.co/internlm/internlm2_5-7b-chat/blob/main/modeling_internlm2.py#L1357) to see how the chat history is organized.

Currently, the InternLM2-Chat model's vocabulary maintains the following mappings to support full functionalities:

- `<|im_start|>`: Start token for each role's dialogue, the token ID is **92543**
- `<|im_end|>`: End token for each role's dialogue, the token ID is **92542**
- `<|action_start|>`: Start token for invoking external tools, like interpreter or plugin, the token ID is **92541**
- `<|action_end|>`: End token for invoking external plugins, the token ID is **92540**
- `<|interpreter|>`: Code interpreter, the token ID is **92539**
- `<|plugin|>`: External plugins, regular tools, the token ID is **92538**

## Complete Structure

The complete chat format of InternLM2-Chat, based on the basic structure, also includes designs for general-purpose AI agents. Its core purpose is to use a streaming format that allows the same format to support various types of plugin extensions and AI environments while being compatible with general chat.

### Function Call

InternLM2-Chat support multiple formats (e.g., ReAct) to conduct function call, especially **json format** to ease downstream applications。An example of complete function call is shown below.

```
<|im_start|>system
You are InternLM2-Chat, a harmless AI assistant<|im_end|>
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
I want to know today's weather in Shanghai<|im_end|>
<|im_start|>assistant
Sure, I will search for the weather of Shanghai.<|action_start|><|plugin|>
{"name": "get_current_weather", "parameters": {"location": "Shanghai"}}<|action_end|><|im_end|>
<|im_start|>environment name=<|plugin|>
{"temperature": 22}<|im_end|>
<|im_start|>assistant
The weather in Shanghai is 22 celsius<|im_end|>
```

1. First，there will be a new system prompt that describe the protocol of tools in json format. The content starts with `<|im_start|>system name=<|plugin|>\n` and ends with `<|im_end|>`. `name=<|plugin|>` indicates the system prompt comes from tools. InternLM2-Chat supports and follows multiple system prompts in the chat history, so we can also see the system prompt ask the model to be helpful assistant.
2. Model will call the tools in a streaming format after receiving the user prompt, i.e., it will naturally speak something (thoughts, helpful response) then call the tools. The model will output `<|action_start|><|plugin|>`, where `<|action_start|>` indicates the model needs to call extensions and `<|plugin|>` indicates it want to use common tools. Then the model will output the calling parameters in json format and end it with `<|action_end|>`.
3. The environment returns the function calling outputs, starts with `<|im_start|>environment name=<|plugin|>`, indicating it comes from the function calling in the environment. The content is in json format, `{"temperature": 22}`, ends with `<|im_end|>`.

### Code Interpreter

InternLM2-Chat support to use code interpreter to solve complex math problems and data analysis. The whole process is similar to the function calling, shown as below.

````
<|im_start|>system
You are InternLM2-Chat, a harmless AI assistant<|im_end|>
<|im_start|>system name=<|interpreter|>
You now have access to a Jupyter notebook environment supporting Python code execution. Just send code to python to run in this stateful environment. This feature is suitable for:
- Data analysis or processing (such as data manipulation and graphic creation)
- Complex calculations (such as math and physics problems)
- Programming examples (for understanding programming concepts or language features)
- Text processing and analysis (including text analysis and natural language processing)
- Machine learning and data science (model training and data visualization)
- File operations and data import (handling CSV, JSON, etc. formats)<|im_end|>
<|im_start|>user
Please help me process and visualize this dataset.
<|im_end|>
<|im_start|>user name=file
[{"path": "data.csv", size='10K'}]<|im_end|>
<|im_start|>assistant
I have processed the data and visualized it for you.

<|action_start|><|interpreter|>
```python
import plotly.express as px
import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv('data.csv')

# Create a scatter plot of rainfall vs wind direction
fig = px.scatter(df, x='WindDir9am', y='Rainfall', color='WindDir3pm',
                 labels={'WindDir9am': 'Wind Direction 9am', 'Rainfall': '

Rainfall', 'WindDir3pm': 'Wind Direction 3pm'},
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
Since the code output is not included here, I cannot provide specific chart content. However, if the code executed correctly, it should display a polar plot with two filled areas representing the relationship between wind direction at 9 am and rainfall, and between wind direction at 3 pm and rainfall, respectively. The values for each direction are based on the average rainfall calculated from the provided dataset. The chart should have a clear title, a legend, and be intuitive for comparing rainfall with different wind directions. Given the use of a dark theme, the overall appearance of the chart should be bright lines and filled areas on a dark background.<|im_end|>
<|im_start|>user
Can you explain the code?<|im_end|>
<|im_start|>assistant
The code uses Python's Plotly library to create a polar plot that shows the relationship between wind direction and rainfall. It starts by creating a Plotly Figure object. Then, I added two `Scatterpolar` traces. The first trace shows the relationship between wind direction at 9 am and average rainfall, and the second trace shows the same for 3 pm. In these traces, the `r` parameter represents the average rainfall, and `theta` represents the wind direction. `fill='toself'` means the shape is filled to its own boundary. Next, I updated the chart layout, hid the radial axis, added a legend, set the title and its font, chose a dark theme, and set the chart's background color, width, and height. Finally, the chart was displayed using `fig.show()`.
<|im_end|>
````

1. First，there will be a new system prompt that describe the usages of code interpreter. The content starts with `<|im_start|>system name=<|interpreter|>\n` and ends with `<|im_end|>`. `name=<|interpreter|>` indicates the system prompt comes from code interpreter. InternLM2-Chat supports and follows multiple system prompts in the chat history, so we can also see the system prompt ask the model to be helpful assistant.
2. Model will call the tools in a streaming format after receiving the user prompt, i.e., it will naturally speak something (thoughts, helpful response) then call the code interpreter. The model will output `<|action_start|><|interpreter|>`, where `<|action_start|>` indicates the model needs to call extensions and `<|interpreter|>` indicates it want to use code interpreter. Then the model will output the code in a markdown-style code block and end it with `<|action_end|>`.
3. The environment will execute the code and return outputs, starts with `<|im_start|>environment name=<|interpreter|>`, indicating it comes from the code interpreter in the environment, and the content ends with `<|im_end|>`.

### Function Call + Code Interpreter

InternLM2-Chat can choose to use tools or code interpreter automatically. We only need to send the system prompt of both to the model respectively. An example is shown as below

````
<|im_start|>system
You are InternLM2-Chat, a harmless AI assistant<|im_end|>
<|im_start|>system name=<|interpreter|>
You now have access to a Jupyter notebook environment supporting Python code execution. Just send code to python to run in this stateful environment. This feature is suitable for:
- Data analysis or processing (such as data manipulation and graphic creation)
- Complex calculations (such as math and physics problems)
- Programming examples (for understanding programming concepts or language features)
- Text processing and analysis (including text analysis and natural language processing)
- Machine learning and data science (model training and data visualization)
- File operations and data import (handling CSV, JSON, etc. formats)<|im_end|>
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
Please help me process and visualize this dataset.
<|im_end|>
<|im_start|>user name=file
[{"path": "data.csv", size='10K'}]<|im_end|>
<|im_start|>assistant
I have processed the data and visualized it for you.

<|action_start|><|interpreter|>
```python
import plotly.express as px
import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv('data.csv')

# Create a scatter plot of rainfall vs wind direction
fig = px.scatter(df, x='WindDir9am', y='Rainfall', color='WindDir3pm',
                 labels={'WindDir9am': 'Wind Direction 9am', 'Rainfall': '

Rainfall', 'WindDir3pm': 'Wind Direction 3pm'},
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
Since the code output is not included here, I cannot provide specific chart content. However, if the code executed correctly, it should display a polar plot with two filled areas representing the relationship between wind direction at 9 am and rainfall, and between wind direction at 3 pm and rainfall, respectively. The values for each direction are based on the average rainfall calculated from the provided dataset. The chart should have a clear title, a legend, and be intuitive for comparing rainfall with different wind directions. Given the use of a dark theme, the overall appearance of the chart should be bright lines and filled areas on a dark background.<|im_end|>
<|im_start|>user
Can you explain the code?<|im_end|>
<|im_start|>assistant
The code uses Python's Plotly library to create a polar plot that shows the relationship between wind direction and rainfall. It starts by creating a Plotly Figure object. Then, I added two `Scatterpolar` traces. The first trace shows the relationship between wind direction at 9 am and average rainfall, and the second trace shows the same for 3 pm. In these traces, the `r` parameter represents the average rainfall, and `theta` represents the wind direction. `fill='toself'` means the shape is filled to its own boundary. Next, I updated the chart layout, hid the radial axis, added a legend, set the title and its font, chose a dark theme, and set the chart's background color, width, and height. Finally, the chart was displayed using `fig.show()`.
<|im_end|>
<|im_start|>user
I want to know today's weather in Shanghai<|im_end|>
<|im_start|>assistant
Sure, I will search for the weather of Shanghai.<|action_start|><|plugin|>
{"name": "get_current_weather", "parameters": {"location": "Shanghai"}}<|action_end|><|im_end|>
<|im_start|>environment name=<|plugin|>
{"temperature": 22}<|im_end|>
<|im_start|>assistant
The weather in Shanghai is 22 celsius<|im_end|>
````
