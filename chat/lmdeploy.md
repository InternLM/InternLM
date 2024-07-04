# Inference by LMDeploy

English | [简体中文](lmdeploy_zh_cn.md)

[LMDeploy](https://github.com/InternLM/lmdeploy) is an efficient, user-friendly toolkit designed for compressing, deploying, and serving LLM models.

This article primarily highlights the basic usage of LMDeploy. For a comprehensive understanding of the toolkit, we invite you to refer to [the tutorials](https://lmdeploy.readthedocs.io/en/latest/).

## Installation

Install lmdeploy with pip (python 3.8+)

```shell
pip install lmdeploy>=0.2.1
```

## Offline batch inference

With just 4 lines of codes, you can execute batch inference using a list of prompts:

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2_5-7b-chat")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

With dynamic ntk, LMDeploy can handle a context length of 200K for `InternLM2`:

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

For more information about LMDeploy pipeline usage, please refer to [here](https://lmdeploy.readthedocs.io/en/latest/inference/pipeline.html).

## Serving

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat
```

The default port of `api_server` is `23333`. After the server is launched, you can communicate with server on terminal through `api_client`:

```shell
lmdeploy serve api_client http://0.0.0.0:23333
```

Alternatively, you can test the server's APIs oneline through the Swagger UI at `http://0.0.0.0:23333`. A detailed overview of the API specification is available [here](https://lmdeploy.readthedocs.io/en/latest/serving/restful_api.html).
