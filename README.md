# InternLM

<div align="center">

<img src="./assets/logo.svg" width="200"/>
  <div> </div>
  <div align="center">
    <b><font size="5">InternLM</font></b>
    <sup>
      <a href="https://internlm.intern-ai.org.cn/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    <div> </div>
  </div>

[![license](./assets/license.svg)](./LICENSE)
[![evaluation](./assets/compass_support.svg)](https://github.com/internLM/OpenCompass/)
<!-- [![Documentation Status](https://readthedocs.org/projects/internlm/badge/?version=latest)](https://internlm.readthedocs.io/zh_CN/latest/?badge=latest) -->
[📘Inference](./inference) |
[🛠️Agent](./agent) |
[📊Evaluation](./evaluation) |
[👀Model](./model_cards) |
[🤗HuggingFace](https://huggingface.co/spaces/internlm/internlm2-Chat-7B) |
[🆕Update News](#news) |
[🤔Reporting Issues](https://github.com/InternLM/InternLM/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">WeChat</a>
</p>

## Introduction

- **Long (200k) context support**: Both base and chat models can work with more than 200K context after being sufficiently trained on 32K-context data. Try it with [LMDeploy](./inference/) for 200K-context inference.

- **Math & code capability**: InternLM2 series has significantly better performance on Math and Code as general base and chat models.

- **Language generation capability**: Better language generation (chat, writing, Chinese poem, etc.) quality due to high-quality training corpus.

- **Agent capability**: State-of-the-art tool utilization capabilities, especially zero-shot tool calling and code interpreter for math and data analysis, in both [streaming](docs/chat_format.md##streaming-style) and [ReAct](docs/chat_format.md##react-style) format. Try it with [Lagent](./agent/).

## News

[2024.01.17] We release InternLM2-7B and InternLM2-20B and their corresponding chat models with stronger capabilities in all dimensions. See [model zoo below](#model-zoo) or [model cards](./model_cards/) for more details.

[2023.12.13] InternLM-7B-Chat and InternLM-20B-Chat checkpoints are updated. With an improved finetuning strategy, the new chat models can generate higher quality responses with greater stylistic diversity.

[2023.09.20] InternLM-20B is released with base and chat versions.

## Model Zoo

| Model | Transformers(HF) | ModelScope(HF) | OpenXLab(HF) | Release Date |
|---------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| **InternLM2 Chat 20B**     | [🤗internlm/internlm-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)         | [<img src="./assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm2-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b/summary)         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b)     | 2024-01-17   |
| **InternLM2 20B** | [🤗internlm/internlm2-20b](https://huggingface.co/internlm/internlm2-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm2-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b) | 2024-01-17 |
| **InternLM2 Chat 20B SFT**     | [🤗internlm/internlm-chat-20b-sft](https://huggingface.co/internlm/internlm2-chat-20b-sft)         | [<img src="./assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm2-chat-20b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b-sft/summary)         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft)     | 2024-01-17   |
| **InternLM2 Base 20B** | [🤗internlm/internlm2-base-20b](https://huggingface.co/internlm/internlm2-base-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm2-base-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b) | 2024-01-17 |
| **InternLM2 Chat 7B**      | [🤗internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)           | [<img src="./assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b)      | 2024-01-17  |
| **InternLM2 7B**           | [🤗internlm/internlm2-7b](https://huggingface.co/internlm/internlm2-7b)                     | [<img src="./assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b/summary)                     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b)           |  2024-01-17   |
| **InternLM2 Chat 7B SFT**      | [🤗internlm/internlm2-chat-7b-sft](https://huggingface.co/internlm/internlm2-chat-7b-sft)           | [<img src="./assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm2-chat-7b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b-sft/summary)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft)      | 2024-01-17  |
| **InternLM2 Base 7B**           | [🤗internlm/internlm2-base-7b](https://huggingface.co/internlm/internlm2-base-7b)                     | [<img src="./assets/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm2-base-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b/summary)                     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b)           |  2024-01-17   |

**Note:**

1. For chat models, InternLM2 Chat 7/20B has gone through online RLHF for better alignment, which is recommended for downstream applications. We also released InternLM2 Chat 7/20B SFT, which are the ones that only have gone through SFT and used in RLHF to obtain InternLM2 Chat 7/20B. InternLM2 Chat 7/20B are trained from InternLM2 Base 7/20B.
2. For base models, InternLM2 7/20B are further trained from InternLM2 Base 7/20B, which is recommended for fast adaptation for downstream applications.

**Limitations:** Although we have made efforts to ensure the safety of the model during the training process and to encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the dissemination of harmful information.

## Usages

We briefly show the usages with [Transformers](#import-from-transformers), [ModelScope](#import-from-modelscope), and [Web demos](#dialogue).
The chat models adopt [chatml format](./chat/chat_format.md) to support both chat and agent applications.

### Import from Transformers

To load the InternLM2 7B Chat model using Transformers, use the following code:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True).cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "hello", history=[])
>>> print(response)
Hello! How can I help you today?
>>> response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
>>> print(response)
Sure, here are three tips for effective time management:

1. Prioritize tasks based on importance and urgency: Make a list of all your tasks and categorize them into "important and urgent," "important but not urgent," and "not important but urgent." Focus on completing the tasks in the first category before moving on to the others.
2. Use a calendar or planner: Write down deadlines and appointments in a calendar or planner so you don't forget them. This will also help you schedule your time more effectively and avoid overbooking yourself.
3. Minimize distractions: Try to eliminate any potential distractions when working on important tasks. Turn off notifications on your phone, close unnecessary tabs on your computer, and find a quiet place to work if possible.

Remember, good time management skills take practice and patience. Start with small steps and gradually incorporate these habits into your daily routine.
```

### Import from ModelScope

To load the InternLM model using ModelScope, use the following code:

```python
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir,device_map="auto",  trust_remote_code=True,torch_dtype=torch.float16)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

### Dialogue

You can interact with the InternLM Chat 7B model through a frontend interface by running the following code:

```bash
pip install streamlit==1.24.0
pip install transformers==4.30.2
streamlit run ./chat/web_demo.py
```

The effect is as follows

![demo](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

### Deployment

We use [LMDeploy](https://github.com/InternLM/LMDeploy) for fast deployment of InternLM.

```shell
# install LMDeploy
python3 -m pip install lmdeploy
# chat with internlm2
lmdeploy chat turbomind InternLM/internlm2-chat-7b --model-name internlm2-chat-7b
```

Please refer to the [guidance](./chat/lmdeploy.md) for more usages about model deployment. For additional deployment tutorials, feel free to explore [here](https://github.com/InternLM/LMDeploy).

## Agent

InternLM2-Chat models have excellent tool utilization capabilities and can work with function calls in a zero-shot manner. See more examples in [agent session](./agent/).

## Fine-tuning

Please refer to [finetune docs](./finetune) for fine-tuning with InternLM.

## Contribution

We appreciate all the contributors for their efforts to improve and enhance InternLM. Community users are highly encouraged to participate in the project. Please refer to the contribution guidelines for instructions on how to contribute to the project.

## License

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [application form (English)](https://wj.qq.com/s2/12727483/5dba/)/[申请表（中文）](https://wj.qq.com/s2/12725412/f7c1/). For other questions or collaborations, please contact <internlm@pjlab.org.cn>.

## Citation

```
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```
