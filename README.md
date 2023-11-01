# InternLM

<div align="center">

<img src="./doc/imgs/logo.svg" width="200"/>
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

[![license](./doc/imgs/license.svg)](./LICENSE)
[![evaluation](./doc/imgs/compass_support.svg)](https://github.com/internLM/OpenCompass/)
[![Documentation Status](https://readthedocs.org/projects/internlm/badge/?version=latest)](https://internlm.readthedocs.io/zh_CN/latest/?badge=latest)

[📘Usage](./doc/en/usage.md) |
[🛠️Installation](./doc/en/install.md) |
[📊Train Performance](./doc/en/train_performance.md) |
[👀Model](#model-zoo) |
[🤗HuggingFace](https://huggingface.co/spaces/internlm/InternLM-Chat-7B) |
[🆕Update News](./CHANGE_LOG.md) |
[🤔Reporting Issues](https://github.com/InternLM/InternLM/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">WeChat</a>
</p>

## Introduction
InternLM is an open-sourced lightweight training framework aims to  support model pre-training without the need for extensive dependencies. With a single codebase, it supports pre-training on large-scale clusters with thousands of GPUs, and fine-tuning on a single GPU while achieving remarkable performance optimizations. InternLM achieves nearly 90% acceleration efficiency during training on 1024 GPUs.

Based on the InternLM training framework, we have released two open-sourced pretrained model InternLM-7B and InternLM-20B.


## News

[20230920] InternLM-20B is released with base and chat versions.  
[20230822] InternLM-7B-Chat v1.1 is released with code interpreter and function calling capability. You can try it with [Lagent](https://github.com/InternLM/lagent).


## Model Zoo

Our models are released in three platforms: Transformers, ModelScope and OpenXLab.  
- There are two kinds of model weights: 
  1. huggingface type(marked as HF)
  2. original model weight(marked as Original), providing in OpenXLab, which can be loaded by InternLM and finetuned directly.

| Model | Transformers(HF) | ModelScope(HF) | OpenXLab(HF) | OpenXLab(Original) | Release Date |
|---------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| **InternLM Chat 20B**     | [🤗internlm/internlm-chat-20b](https://huggingface.co/internlm/internlm-20b-chat)         | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b-chat/summary)         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-20b)     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-20b-original)     | 2023-09-20   |
| **InternLM 20B** | [🤗internlm/internlm-20b](https://huggingface.co/internlm/internlm-20b) | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-20b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-20b-original) | 2023-09-20 |
| **InternLM Chat 7B v1.1** | [🤗internlm/internlm-chat-7b-v1.1](https://huggingface.co/internlm/internlm-chat-7b-v1.1) | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-7b-v1_1](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-v1_1/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-v1.1) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-v1.1-original) | 2023-08-22   |
| **InternLM 7B**           | [🤗internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b)                     | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary)                     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-7b)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-7b-original)           | 2023-07-06   |
| **InternLM Chat 7B**      | [🤗internlm/internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)           | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b/summary)           | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b)      | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-original)      | 2023-07-06   |
| **InternLM Chat 7B 8k**   | [🤗internlm/internlm-chat-7b-8k](https://huggingface.co/internlm/internlm-chat-7b-8k)     | [<img src="./doc/imgs/modelscope_logo.png" width="20px" /> Shanghai_AI_Laboratory/internlm-chat-7b-8k](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-8k/summary)     | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-8k)   | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-8k-original)   | 2023-07-06   |

#### Introduction
InternLM-20B was pre-trained on over **2.3T** Tokens containing high-quality English, Chinese, and code data. Additionally, the Chat version has undergone SFT and RLHF training, enabling it to better and more securely meet users' needs.

In terms of model structure, InternLM-20B opted for a deeper architecture, with a depth set at 60 layers. This surpasses the conventional 7B and 13B models that utilize 32 or 40 layers. When parameters are limited, increasing the number of layers can enhance the model's overall capability. Furthermore, compared to InternLM-7B, the pre-training data used for InternLM-20B underwent higher quality cleansing and was supplemented with data rich in knowledge and designed for reinforcing understanding and reasoning capabilities. As a result, it exhibits significant improvements in understanding, reasoning, mathematical, and programming abilities—all of which test the technical proficiency of language models. Overall, InternLM-20B features the following characteristics:
- Outstanding overall performance
- Strong utility invocation capability
- Supports a 16k context length (Through inference extrapolation)
- Better value alignment.

#### Performance Evaluation

On the 5 capability dimensions proposed by OpenCompass, InternLM-20B has achieved excellent results (the bolded scores represent the best performances within the 13B-33B parameter range).

| Capability | Llama-13B | Llama2-13B | Baichuan2-13B | InternLM-20B | Llama-33B | Llama-65B | Llama2-70B |
|----------|-----------|------------|---------------|--------------|-----------|-----------|------------|
| Language     | 42.5      | 47         | 47.5          | **55**           | 44.6      | 47.1      | 51.6       |
| Knowledge     | 58.2      | 58.3       | 48.9          | 60.1         | **64**        | 66        | 67.7       |
| Understanding     | 45.5      | 50.9       | 58.1          | **67.3**         | 50.6      | 54.2      | 60.8       |
| Reasoning     | 42.7      | 43.6       | 44.2          | **54.9**         | 46.4      | 49.8      | 55         |
| Examination     | 37.3      | 45.2       | 51.8          | **62.5**         | 47.4      | 49.7      | 57.3       |
| Overall   | 43.8      | 47.3       | 49.4          | **59.2**         | 48.9      | 51.9      | 57.4       |

The table below compares the performance of mainstream open-source models on some influential and typical datasets.

|      | Benchmarks           | Llama-13B | Llama2-13B | Baichuan2-13B | InternLM-20B | Llama-33B | Llama-65B | Llama2-70B |
|------|------------------|-----------|------------|---------------|--------------|-----------|-----------|------------|
| Examination | MMLU             | 47.73     | 54.99      | 59.55         | **62.05**        | 58.73     | 63.71     | 69.75      |
|      | C-Eval (val)     | 31.83     | 41.4       | **59.01**         | 58.8         | 37.47     | 40.36     | 50.13      |
|      | AGI-Eval         | 22.03     | 30.93      | 37.37         | **44.58**        | 33.53     | 33.92     | 40.02      |
| Knowledge | BoolQ            | 78.75     | 82.42      | 67            | **87.46**        | 84.43     | 86.61     | 87.74      |
|      | TriviaQA         | 52.47     | 59.36      | 46.61         | 57.26        | **66.24**     | 69.79     | 70.71      |
|      | NaturalQuestions | 20.17     | 24.85      | 16.32         | 25.15        | **30.89**     | 33.41     | 34.16      |
| Understanding | CMRC             | 9.26      | 31.59      | 29.85         | **68.78**        | 14.17     | 34.73     | 43.74      |
|      | CSL              | 55        | 58.75      | 63.12         | **65.62**        | 57.5      | 59.38     | 60         |
|      | RACE (middle)    | 53.41     | 63.02      | 68.94         | **86.35**        | 64.55     | 72.35     | 81.55      |
|      | RACE (high)      | 47.63     | 58.86      | 67.18         | **83.28**        | 62.61     | 68.01     | 79.93      |
|      | XSum             | 20.37     | 23.37      | 25.23         | **35.54**        | 20.55     | 19.91     | 25.38      |
| Reasoning | WinoGrande       | 64.64     | 64.01      | 67.32         | **69.38**        | 66.85     | 69.38     | 69.77      |
|      | BBH              | 37.93     | 45.62      | 48.98         | **52.51**        | 49.98     | 58.38     | 64.91      |
|      | GSM8K            | 20.32     | 29.57      | **52.62**         | **52.62**        | 42.3      | 54.44     | 63.31      |
|      | PIQA             | 79.71     | 79.76      | 78.07         | 80.25        | **81.34**     | 82.15     | 82.54      |
| Programming | HumanEval        | 14.02     | 18.9       | 17.07         | **25.61**        | 17.68     | 18.9      | 26.22      |
|      | MBPP             | 20.6      | 26.8       | 30.8          | **35.6**         | 28.4      | 33.6      | 39.6       |

Overall, InternLM-20B comprehensively outperforms open-source models in the 13B parameter range in terms of overall capabilities, and on inference evaluation sets, it approaches or even surpasses the performance of Llama-65B.

- The evaluation results were obtained from [OpenCompass 20230920](https://github.com/internLM/OpenCompass/).
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/internLM/OpenCompass/), so please refer to the latest evaluation results of [OpenCompass](https://github.com/internLM/OpenCompass/).

</details>


<details> 
<summary> InternLM-7B </summary>

#### News
[20230822] By utilizing richer SFT-type data, the InternLM-7B-Chat v1.1 model supports code interpretation and function invocation. The model structure and code remain unchanged, so the more powerful InternLM-7B-Chat v1.1 can be used in exactly the same way as InternLM-7B-Chat.

#### Introduction
InternLM-7B contains a 7 billion parameter base model and a chat model tailored for practical scenarios. The model has the following characteristics:

- It leverages trillions of high-quality tokens for training to establish a powerful knowledge base.
- It supports an 8k context window length, enabling longer input sequences and stronger reasoning capabilities.
- It provides a versatile toolset for users to flexibly build their own workflows.

#### Performance Evaluation

We conducted a comprehensive evaluation of InternLM using the open-source evaluation tool [OpenCompass](https://github.com/internLM/OpenCompass/). The evaluation covered five dimensions of capabilities: disciplinary competence, language competence, knowledge competence, inference competence, and comprehension competence. Here are some of the evaluation results, and you can visit the [OpenCompass leaderboard](https://opencompass.org.cn/rank) for more evaluation results.

| Datasets\Models | **InternLM-Chat-7B** | **InternLM-7B** | LLaMA-7B | Baichuan-7B | ChatGLM2-6B | Alpaca-7B | Vicuna-7B |
| --------------- | -------------------------- | --------------------- | -------- | ----------- | ----------- | --------- | --------- |
| C-Eval(Val)     | 53.2                       | 53.4                  | 24.2     | 42.7        | 50.9        | 28.9      | 31.2      |
| MMLU            | 50.8                       | 51.0                  | 35.2*    | 41.5        | 46.0        | 39.7      | 47.3      |
| AGIEval         | 42.5                       | 37.6                  | 20.8     | 24.6        | 39.0        | 24.1      | 26.4      |
| CommonSenseQA   | 75.2                       | 59.5                  | 65.0     | 58.8        | 60.0        | 68.7      | 66.7      |
| BUSTM           | 74.3                       | 50.6                  | 48.5     | 51.3        | 55.0        | 48.8      | 62.5      |
| CLUEWSC         | 78.6                       | 59.1                  | 50.3     | 52.8        | 59.8        | 50.3      | 52.2      |
| MATH            | 6.4                        | 7.1                   | 2.8      | 3.0         | 6.6         | 2.2       | 2.8       |
| GSM8K           | 34.5                       | 31.2                  | 10.1     | 9.7         | 29.2        | 6.0       | 15.3      |
| HumanEval       | 14.0                       | 10.4                  | 14.0     | 9.2         | 9.2         | 9.2       | 11.0      |
| RACE(High)      | 76.3                       | 57.4                  | 46.9*    | 28.1        | 66.3        | 40.7      | 54.0      |

- The evaluation results were obtained from [OpenCompass 20230706](https://github.com/internLM/OpenCompass/) (some data marked with *, which means come from the original papers), and evaluation configuration can be found in the configuration files provided by [OpenCompass](https://github.com/internLM/OpenCompass/).
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/internLM/OpenCompass/), so please refer to the latest evaluation results of [OpenCompass](https://github.com/internLM/OpenCompass/).

</details>

**Limitations:** Although we have made efforts to ensure the safety of the model during the training process and to encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the dissemination of harmful information.

## Usage Examples

### Import from Transformers

To load the InternLM 7B Chat model using Transformers, use the following code:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b-v1_1", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b-v1_1", trust_remote_code=True).cuda()
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
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b-v1_1', revision='v1.0.0')
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
streamlit run web_demo.py
```

The effect is as follows

![demo](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

### Deployment

We use [LMDeploy](https://github.com/InternLM/LMDeploy) to complete the one-click deployment of InternLM.

1. First, install LMDeploy:

```
  python3 -m pip install lmdeploy
```

2. Use the following command for quick deployment:

```
  python3 -m lmdeploy.serve.turbomind.deploy InternLM-7B /path/to/internlm-7b/model hf
```

3. After exporting the model, you can start a server and have a conversation with the deployed model using the following command:

```
  python3 -m lmdeploy.serve.client {server_ip_addresss}:33337
```

[LMDeploy](https://github.com/InternLM/LMDeploy) provides a complete workflow for deploying InternLM. Please refer to the [deployment tutorial](https://github.com/InternLM/LMDeploy) for more details on deploying InternLM.

## Fine-tuning & Training

### Pre-training and Fine-tuning Tutorial

Please refer to [Usage Tutorial](./doc/en/usage.md) to start InternLM installation, data processing, pre-training and fine-tuning.

### Convert to Transformers Format

The model trained by InternLM can be easily converted to HuggingFace Transformers format, which is convenient for seamless docking with various open source projects in the community. With the help of `tools/transformers/convert2hf.py`, the weights saved during training can be converted into transformers format with one command

```bash
python tools/transformers/convert2hf.py --src_folder origin_ckpt/ --tgt_folder hf_ckpt/ --tokenizer ./tools/V7_sft.model
```

After conversion, it can be loaded as transformers by the following code

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

## Training System

### System Architecture

Please refer to the [System Architecture document](./doc/en/structure.md) for further details.

### Training Performance

InternLM deeply integrates Flash-Attention, Apex and other high-performance model operators to improve training efficiency. By building the Hybrid Zero technique, it achieves efficient overlap of computation and communication, significantly reducing cross-node communication traffic during training. InternLM supports expanding the 7B model from 8 GPUs to 1024 GPUs, with an acceleration efficiency of up to 90% at the thousand-GPU scale, a training throughput of over 180 TFLOPS, and an average of over 3600 tokens per GPU per second. The following table shows InternLM's scalability test data at different configurations:

| GPU Number         | 8   | 16  | 32  | 64  | 128  | 256  | 512  | 1024  |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| TGS | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| TFLOPS  | 193 | 191  | 188  | 188  | 187   | 185   | 186   | 184    |

TGS represents the average number of tokens processed per GPU per second. For more performance test data, please refer to the [Training Performance document](./doc/en/train_performance.md) for further details.

## Contribution

We appreciate all the contributors for their efforts to improve and enhance InternLM. Community users are highly encouraged to participate in the project. Please refer to the contribution guidelines for instructions on how to contribute to the project.

## Acknowledgements

InternLM codebase is an open-source project contributed by Shanghai AI Laboratory and researchers from different universities and companies. We would like to thank all the contributors for their support in adding new features to the project and the users for providing valuable feedback. We hope that this toolkit and benchmark can provide the community with flexible and efficient code tools for fine-tuning InternLM and developing their own models, thus continuously contributing to the open-source community. Special thanks to the two open-source projects, [flash-attention](https://github.com/HazyResearch/flash-attention) and [ColossalAI](https://github.com/hpcaitech/ColossalAI).

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
