# LMDeploy 推理

[English](lmdeploy.md) | 简体中文

[LMDeploy](https://github.com/InternLM/lmdeploy) 是一个高效且友好的 LLM 模型部署工具箱，功能涵盖了量化、推理和服务。

本文主要介绍 LMDeploy 的基本用法，包括[安装](#安装)、[离线批处理](#离线批处理)和[推理服务](#推理服务)。更全面的介绍请参考 [LMDeploy 用户指南](https://lmdeploy.readthedocs.io/zh-cn/latest/)。

## 安装

使用 pip（python 3.8+）安装 LMDeploy

```shell
pip install lmdeploy>=0.2.1
```

## 离线批处理

只用以下 4 行代码，就可以完成 prompts 的批处理：

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2_5-7b-chat")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

LMDeploy 实现了 dynamic ntk，支持长文本外推。使用如下代码，可以把 InternLM2 的文本外推到 200K：

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig(session_len=200000,
                                      rope_scaling_factor=2.0)
pipe = pipeline("internlm/internlm2_5-7b-chat", backend_engine=engine_config)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
response = pipe(prompt, gen_config=gen_config)
print(response)
```

更多关于 pipeline 的使用方式，请参考[这里](https://lmdeploy.readthedocs.io/zh-cn/latest/inference/pipeline.html)

## 推理服务

LMDeploy `api_server` 支持把模型一键封装为服务，对外提供的 RESTful API 兼容 openai 的接口。以下为服务启动的示例：

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat
```

服务默认端口是23333。在 server 启动后，你可以在终端通过`api_client`与server进行对话，体验对话效果：

```shell
lmdeploy serve api_client http://0.0.0.0:23333
```

此外，你还可以通过 Swagger UI `http://0.0.0.0:23333` 在线阅读和试用 `api_server` 的各接口，也可直接查阅[文档](https://lmdeploy.readthedocs.io/zh-cn/latest/serving/restful_api.html)，了解各接口的定义和使用方法。
