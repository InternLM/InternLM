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
[📘Chat](./chat) |
[🛠️Agent](./agent) |
[📊Evaluation](#evaluation) |
[👀Model](./model_cards) |
[🤗HuggingFace](https://huggingface.co/spaces/internlm/internlm2-Chat-7B) |
[🆕Update News](#news) |
[🤔Reporting Issues](https://github.com/InternLM/InternLM/issues/new)

[English](./README.md) |
[简体中文](./README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">WeChat</a>
</p>

## Introduction

InternLM2 series are released with the following features:

- **200K Context window**: Nearly perfect at finding needles in the haystack with 200K-long context, with leading performance on long-context tasks like LongBench and L-Eval. Try it with [LMDeploy](./inference/) for 200K-context inference.

- **Outstanding comprehensive performance**: Significantly better than the last generation in all dimensions, especially in reasoning, math, code, chat experience, instruction following, and creative writing, with leading performance among open-source models in similar sizes. In some evaluations, InternLM2-Chat-20B may match or even surpass ChatGPT (GPT-3.5).

- **Code interpreter & Data analysis**: With code interpreter, InternLM2-Chat-20B obtains compatible performance with GPT-4 on GSM8K and MATH. InternLM2-Chat also provides data analysis capability.

- **Stronger tool use**: Based on better tool utilization-related capabilities in instruction following, tool selection and reflection, InternLM2 can support more kinds of agents and multi-step tool calling for complex tasks. See [examples](./agent/).

## News

[2024.01.17] We release InternLM2-7B and InternLM2-20B and their corresponding chat models with stronger capabilities in all dimensions. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/) for more details.

[2023.12.13] InternLM-7B-Chat and InternLM-20B-Chat checkpoints are updated. With an improved finetuning strategy, the new chat models can generate higher quality responses with greater stylistic diversity.

[2023.09.20] InternLM-20B is released with base and chat versions.

## Model Zoo

| Model | Transformers(HF) | ModelScope(HF) | OpenXLab(HF) | Release Date |
|---------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| **InternLM2 Chat 20B**     | [🤗internlm/internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)         | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b/summary)         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b)     | 2024-01-17   |
| **InternLM2 20B** | [🤗internlm/internlm2-20b](https://huggingface.co/internlm/internlm2-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b) | 2024-01-17 |
| **InternLM2 Chat 20B SFT**     | [🤗internlm/internlm2-chat-20b-sft](https://huggingface.co/internlm/internlm2-chat-20b-sft)         | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b-sft/summary)         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft)     | 2024-01-17   |
| **InternLM2 Base 20B** | [🤗internlm/internlm2-base-20b](https://huggingface.co/internlm/internlm2-base-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b) | 2024-01-17 |
| **InternLM2 Chat 7B**      | [🤗internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)           | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b)      | 2024-01-17  |
| **InternLM2 7B**           | [🤗internlm/internlm2-7b](https://huggingface.co/internlm/internlm2-7b)                     | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b/summary)                     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b)           |  2024-01-17   |
| **InternLM2 Chat 7B SFT**      | [🤗internlm/internlm2-chat-7b-sft](https://huggingface.co/internlm/internlm2-chat-7b-sft)           | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b-sft/summary)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft)      | 2024-01-17  |
| **InternLM2 Base 7B**           | [🤗internlm/internlm2-base-7b](https://huggingface.co/internlm/internlm2-base-7b)                     | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b/summary)                     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b)           |  2024-01-17   |

**Note of Models:**

The release of InternLM2 series contains two model sizes: 7B and 20B. 7B models are efficient for research and application and 20B models are more powerful and can support more complex scenarios. For each model size, there are three types of models for different user requirements

1. InternLM2-Base: Foundation models with high quality and high adaptation flexibility, which serves as a good starting point for downstream deep adaptations.
2. InternLM2: Optimized in multiple dimensions based on InternLM2-Base, obtaining state-of-the-art performance in evaluation with good language capability. InternLM2 models are recommended for consideration in most applications.
3. InternLM2-Chat: InternLM2-Chat have gone through SFT and online RLHF based on InternLM2-Base model, for better instruction following, chat experience and function calling, which is recommended for downstream applications. We also released their corresponding SFT version, termed InternLM2 Chat 7/20B SFT, to ease the research for alignment.

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

The effect is similar to below:

![demo](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

### Deployment

We use [LMDeploy](https://github.com/InternLM/LMDeploy) for fast deployment of InternLM.

With only 4 lines of codes, you can perform `internlm2-chat-7b` inference after `pip install lmdeploy`.

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2-chat-7b")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

Please refer to the [guidance](./chat/lmdeploy.md) for more usages about model deployment. For additional deployment tutorials, feel free to explore [here](https://github.com/InternLM/LMDeploy).

## Agent

InternLM2-Chat models have excellent tool utilization capabilities and can work with function calls in a zero-shot manner. See more examples in [agent session](./agent/).

## Fine-tuning

Please refer to [finetune docs](./finetune/) for fine-tuning with InternLM.

**Note:** We have migrated the whole training functionality in this project to [InternEvo](https://github.com/InternLM/InternEvo) for easier user experience, which provides efficient pre-training and fine-tuning infra for training InternLM.


## Evaluation

We utilize [OpenCompass](https://github.com/open-compass/opencompass) for model evaluation. In InternLM-2, we primarily focus on standard objective evaluation, long-context evaluation (needle in a haystack), data contamination assessment, agent evaluation, and subjective evaluation.

### Objective Evaluation

To evaluate the InternLM model, please follow the guidelines in the [OpenCompass tutorial](https://github.com/open-compass/opencompass). Typically, we use `ppl` for multiple-choice questions on the **Base** model and `gen` for all questions on the **Chat** model.

### Long-Context Evaluation (Needle in a Haystack)

For the `Needle in a Haystack` evaluation, refer to the tutorial provided in the [documentation](https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/needleinahaystack_eval.md). Feel free to try it out.

### Data Contamination Assessment

To learn more about data contamination assessment, please check the [contamination eval](https://opencompass.readthedocs.io/en/latest/advanced_guides/contamination_eval.html).

### Agent Evaluation

- To evaluate tool utilization, please refer to [T-Eval](https://github.com/open-compass/T-Eval).
- For code interpreter evaluation, use the [gsm-8k-agent](https://github.com/open-compass/opencompass/blob/main/configs/datasets/gsm8k/gsm8k_agent_gen_be1606.py) provided in the repository. Additionally, you need to install [Lagent](https://github.com/InternLM/lagent).

### Subjective Evaluation

- Please follow the [tutorial](https://opencompass.readthedocs.io/en/latest/advanced_guides/subjective_evaluation.html) for subjective evaluation.

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
