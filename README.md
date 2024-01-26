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

[📘Commercial Application](#license) |
[🤗HuggingFace](https://huggingface.co/internlm) |
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

- **200K Context window**: Nearly perfect at finding needles in the haystack with 200K-long context, with leading performance on long-context tasks like LongBench and L-Eval. Try it with [LMDeploy](./chat/lmdeploy.md) for 200K-context inference.

- **Outstanding comprehensive performance**: Significantly better than the last generation in all dimensions, especially in reasoning, math, code, chat experience, instruction following, and creative writing, with leading performance among open-source models in similar sizes. In some evaluations, InternLM2-Chat-20B may match or even surpass ChatGPT (GPT-3.5).

- **Code interpreter & Data analysis**: With code interpreter, InternLM2-Chat-20B obtains compatible performance with GPT-4 on GSM8K and MATH. InternLM2-Chat also provides data analysis capability.

- **Stronger tool use**: Based on better tool utilization-related capabilities in instruction following, tool selection and reflection, InternLM2 can support more kinds of agents and multi-step tool calling for complex tasks. See [examples](./agent/).

## News

\[2024.01.23\] We release InternLM2-Math-7B and InternLM2-Math-20B with pretraining and SFT checkpoints. They surpass ChatGPT with small sizes. See [InternLM-Math](https://github.com/InternLM/internlm-math) for details and download.

\[2024.01.17\] We release InternLM2-7B and InternLM2-20B and their corresponding chat models with stronger capabilities in all dimensions. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/) for more details.

\[2023.12.13\] InternLM-7B-Chat and InternLM-20B-Chat checkpoints are updated. With an improved finetuning strategy, the new chat models can generate higher quality responses with greater stylistic diversity.

\[2023.09.20\] InternLM-20B is released with base and chat versions.

## Model Zoo

| Model                      | Transformers(HF)                           | ModelScope(HF)                           | OpenXLab(HF)                           | OpenXLab(Origin)                           | Release Date |
| -------------------------- | ------------------------------------------ | ---------------------------------------- | -------------------------------------- | ------------------------------------------ | ------------ |
| **InternLM2-Base-7B**      | [🤗internlm2-base-7b](https://huggingface.co/internlm/internlm2-base-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-7b-original) | 2024-01-17   |
| **InternLM2-7B**           | [🤗internlm2-7b](https://huggingface.co/internlm/internlm2-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b-original) | 2024-01-17   |
| **InternLM2-Chat-7B-SFT**  | [🤗internlm2-chat-7b-sft](https://huggingface.co/internlm/internlm2-chat-7b-sft) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b-sft/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft-original) | 2024-01-17   |
| **InternLM2-Chat-7B**      | [🤗internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-original) | 2024-01-17   |
| **InternLM2-Base-20B**     | [🤗internlm2-base-20b](https://huggingface.co/internlm/internlm2-base-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-base-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-base-20b-original) | 2024-01-17   |
| **InternLM2-20B**          | [🤗internlm2-20b](https://huggingface.co/internlm/internlm2-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b-original) | 2024-01-17   |
| **InternLM2-Chat-20B-SFT** | [🤗internlm2-chat-20b-sft](https://huggingface.co/internlm/internlm2-chat-20b-sft) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b-sft/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-sft-original) | 2024-01-17   |
| **InternLM2-Chat-20B**     | [🤗internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b) | [<img src="./assets/modelscope_logo.png" width="20px" /> internlm2-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-20b-original) | 2024-01-17   |

**Notes:**

The release of InternLM2 series contains two model sizes: 7B and 20B. 7B models are efficient for research and application and 20B models are more powerful and can support more complex scenarios. The relation of these models are shown as follows.

![](https://internlm.oss-cn-shanghai.aliyuncs.com/series.png)

1. **InternLM2-Base**: Foundation models with high quality and high adaptation flexibility, which serve as a good starting point for downstream deep adaptations.
2. **InternLM2**: Further pretrain with general domain data and domain-enhanced corpus, obtaining state-of-the-art performance in evaluation with good language capability. InternLM2 models are recommended for consideration in most applications.
3. **InternLM2-Chat-SFT**: Intermediate version of InternLM2-Chat that only undergoes supervised fine-tuning (SFT), based on the InternLM2-Base model. We release them to benefit research on alignment.
4. **InternLM2-Chat**: Further aligned on top of InternLM2-Chat-SFT through online RLHF. InternLM2-Chat exhibits better instruction following, chat experience, and function call, which is recommended for downstream applications.

**Limitations:** Although we have made efforts to ensure the safety of the model during the training process and to encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the dissemination of harmful information.

**Supplements:** `HF` refers to the format used by HuggingFace in [transformers](https://github.com/huggingface/transformers), whereas `Origin` denotes the format adopted by the InternLM team in [InternEvo](https://github.com/InternLM/InternEvo).

## Performance

### Objective Evaluation

| Dataset          | Baichuan2-7B-Chat | Mistral-7B-Instruct-v0.2 | Qwen-7B-Chat | InternLM2-Chat-7B | ChatGLM3-6B | Baichuan2-13B-Chat | Mixtral-8x7B-Instruct-v0.1 | Qwen-14B-Chat | InternLM2-Chat-20B |
| ---------------- | ----------------- | ------------------------ | ------------ | ----------------- | ----------- | ------------------ | -------------------------- | ------------- | ------------------ |
| MMLU             | 50.1              | 59.2                     | 57.1         | 63.7              | 58.0        | 56.6               | 70.3                       | 66.7          | 66.5               |
| CMMLU            | 53.4              | 42.0                     | 57.9         | 63.0              | 57.8        | 54.8               | 50.6                       | 68.1          | 65.1               |
| AGIEval          | 35.3              | 34.5                     | 39.7         | 47.2              | 44.2        | 40.0               | 41.7                       | 46.5          | 50.3               |
| C-Eval           | 53.9              | 42.4                     | 59.8         | 60.8              | 59.1        | 56.3               | 54.0                       | 71.5          | 63.0               |
| TrivialQA        | 37.6              | 35.0                     | 46.1         | 50.8              | 38.1        | 40.3               | 57.7                       | 54.5          | 53.9               |
| NaturalQuestions | 12.8              | 8.1                      | 18.6         | 24.1              | 14.0        | 12.7               | 22.5                       | 22.9          | 25.9               |
| C3               | 78.5              | 66.9                     | 84.4         | 91.5              | 79.3        | 84.4               | 82.1                       | 91.5          | 93.5               |
| CMRC             | 8.1               | 5.6                      | 14.6         | 63.8              | 43.2        | 27.8               | 5.3                        | 13.0          | 50.4               |
| WinoGrande       | 49.9              | 50.8                     | 54.2         | 65.8              | 61.7        | 50.9               | 60.9                       | 55.7          | 74.8               |
| BBH              | 35.9              | 46.5                     | 45.5         | 61.2              | 56.0        | 42.5               | 57.3                       | 55.8          | 68.3               |
| GSM-8K           | 32.4              | 48.3                     | 44.1         | 70.7              | 53.8        | 56.0               | 71.7                       | 57.7          | 79.6               |
| Math             | 5.7               | 8.6                      | 12.0         | 23.0              | 20.4        | 4.3                | 22.5                       | 27.6          | 31.9               |
| HumanEval        | 17.7              | 35.4                     | 36.0         | 59.8              | 52.4        | 19.5               | 37.8                       | 40.9          | 67.1               |
| MBPP             | 37.7              | 25.7                     | 33.9         | 51.4              | 55.6        | 40.9               | 40.9                       | 30.0          | 65.8               |

- Performance of MBPP is reported with MBPP(Sanitized)

### Alignment Evaluation

- We have evaluated our model on [AlpacaEval 2.0](https://tatsu-lab.github.io/alpaca_eval/) and InternLM2-Chat-20B surpass Claude 2, GPT-4(0613) and Gemini Pro.

| Model Name         | Win Rate | Length |
| ------------------ | -------- | ------ |
| GPT-4 Turbo        | 50.00%   | 2049   |
| GPT-4              | 23.58%   | 1365   |
| GPT-4 0314         | 22.07%   | 1371   |
| Mistral Medium     | 21.86%   | 1500   |
| XwinLM 70b V0.1    | 21.81%   | 1775   |
| InternLM2 Chat 20B | 21.75%   | 2373   |
| Mixtral 8x7B v0.1  | 18.26%   | 1465   |
| Claude 2           | 17.19%   | 1069   |
| Gemini Pro         | 16.85%   | 1315   |
| GPT-4 0613         | 15.76%   | 1140   |
| Claude 2.1         | 15.73%   | 1096   |

- According to the released performance of 2024-01-17.

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0 (2.0.0 and above are recommended)
- Transformers >= 4.34

## Usages

We briefly show the usages with [Transformers](#import-from-transformers), [ModelScope](#import-from-modelscope), and [Web demos](#dialogue).
The chat models adopt [chatml format](./chat/chat_format.md) to support both chat and agent applications.
To ensure a better usage effect, please make sure that the installed transformers library version meets the following requirements before performing inference with [Transformers](#import-from-transformers) or [ModelScope](#import-from-modelscope):

```
transformers >= 4.34
```

### Import from Transformers

To load the InternLM2-7B-Chat model using Transformers, use the following code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM 7B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
# Output: Hello? How can I help you today?
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

### Import from ModelScope

To load the InternLM2-7B-Chat model using ModelScope, use the following code:

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM 7B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

### Dialogue

You can interact with the InternLM Chat 7B model through a frontend interface by running the following code:

```bash
pip install streamlit
pip install transformers>=4.34
streamlit run ./chat/web_demo.py
```

### Deployment

We use [LMDeploy](https://github.com/InternLM/LMDeploy) for fast deployment of InternLM.

With only 4 lines of codes, you can perform `internlm2-chat-7b` inference after `pip install lmdeploy>=0.2.1`.

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

To evaluate the InternLM model, please follow the guidelines in the [OpenCompass tutorial](https://opencompass.readthedocs.io/en/latest/get_started/installation.html). Typically, we use `ppl` for multiple-choice questions on the **Base** model and `gen` for all questions on the **Chat** model.

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
