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
    👋 join us on <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

## Introduction

InternLM has open-sourced a 7 billion parameter base model and a chat model tailored for practical scenarios. The model has the following characteristics:

- It leverages trillions of high-quality tokens for training to establish a powerful knowledge base.
- It supports an 8k context window length, enabling longer input sequences and stronger reasoning capabilities.
- It provides a versatile toolset for users to flexibly build their own workflows.

Additionally, a lightweight training framework is offered to support model pre-training without the need for extensive dependencies. With a single codebase, it supports pre-training on large-scale clusters with thousands of GPUs, and fine-tuning on a single GPU while achieving remarkable performance optimizations. InternLM achieves nearly 90% acceleration efficiency during training on 1024 GPUs.

## News

InternLM-7B-Chat v1.1 is released with code interpreter and function calling capability. You can try it with [Lagent](https://github.com/InternLM/lagent).

## InternLM-7B

### Performance Evaluation

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

### Model Zoo

InternLM 7B and InternLM 7B Chat, trained using InternLM, have been open-sourced. We provide two formats of model weights for use. In addition to loading the models using the Transformers format, you can also load the weights directly using InternLM for further pre-training or human preference alignment training.

| Model                         | InternLM Format Weight Download Link                                                                                                                 | Transformers Format Weight Download Link                                         |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **InternLM 7B**         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-7b)         | [🤗internlm/intern-7b](https://huggingface.co/internlm/internlm-7b)                 |
| **InternLM Chat 7B v1.1**    | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-v1.1)    | [🤗internlm/intern-chat-7b-v1.1](https://huggingface.co/internlm/internlm-chat-7b-v1.1)       |
| **InternLM Chat 7B**    | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b)    | [🤗internlm/intern-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)       |
| **InternLM Chat 7B 8k** | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-8k) | [🤗internlm/intern-chat-7b-8k](https://huggingface.co/internlm/internlm-chat-7b-8k) |

**Limitations:** Although we have made efforts to ensure the safety of the model during the training process and to encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the dissemination of harmful information.

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

We use [LMDeploy](https://github.com/InternLM/LMDeploy) to complete the workflow of InternLM deployment.

```bash
python3 -m pip install lmdeploy
```

You can utilize the following commands to conduct `internlm-chat-7b` FP16 inference, serve it and interact with AI assistant via WebUI:

```bash
# convert weight layout
python3 -m lmdeploy.serve.turbomind.deploy internlm-chat-7b

# inference lmdeploy's turbomind engine
python3 -m lmdeploy.turbomind.chat ./workspace

# serving with gradio
python3 -m lmdeploy.serve.gradio.app ./workspace
```

You can also deploy 4-bit quantized `internlm-chat-7b` model via LMDeploy. It greatly trims down the model's memory overhead to 6G, just 40% of what FP16 inference would take. More importantly, with extreme optimized kernel, the inference performance achieves 2.4x faster than FP16 inference on A100-80G.

Try the followings to enjoy 4-bit `internlm-chat-7b` on a Geforce RTX 30x GPU card. You can find the inference benchmark from [here](https://github.com/InternLM/lmdeploy/blob/main/docs/en/w4a16.md#inference-performance).

```bash
# download prequnantized internlm-chat-7b model from huggingface
git-lfs install
git clone https://huggingface.co/lmdeploy/llama2-chat-7b-w4

# Convert the model's layout and store it in the default path, ./workspace.
python3 -m lmdeploy.serve.turbomind.deploy internlm-chat-7b ./llama2-chat-7b-w4 awq --group-size 128

# inference lmdeploy's turbomind engine
python3 -m lmdeploy.turbomind.chat ./workspace

# serving with gradio
python3 -m lmdeploy.serve.gradio.app ./workspace
```

LMDeploy is an efficient toolkit for compressing, deploying, and serving LLM models. Please refer to the [deployment tutorial](https://github.com/InternLM/LMDeploy) for more details on deploying InternLM.

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
