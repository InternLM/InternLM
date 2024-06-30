# Multi-Chats by OpenAOE

English | [简体中文](openaoe_zh_cn.md)

## Introduction

[OpenAOE](https://github.com/InternLM/OpenAOE) is a LLM-Group-Chat Framework, which can chat with multiple LLMs (commercial/open source LLMs) at the same time. OpenAOE provides both backend API and WEB-UI to meet different usage needs.

Currently already supported LLMs: [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat), [IntenLM-Chat-7B](https://huggingface.co/internlm/internlm-chat-7b), GPT-3.5, GPT-4, Google PaLM, MiniMax, Claude, Spark, etc.

## Quick Run

> \[!TIP\]
> Require python >= 3.9

We provide three different ways to run OpenAOE: `run by pip`， `run by docker` and `run by source code` as well.

### Run by pip

#### **Install**

```shell
pip install -U openaoe
```

#### **Start**

```shell
openaoe -f /path/to/your/config-template.yaml
```

### Run by docker

#### **Install**

There are two ways to get the OpenAOE docker image by:

1. pull the OpenAOE docker image

```shell
docker pull opensealion/openaoe:latest
```

2. or build a docker image

```shell
git clone https://github.com/internlm/OpenAOE
cd OpenAOE
docker build . -f docker/Dockerfile -t openaoe:latest
```

#### **Start**

```shell
docker run -p 10099:10099 -v /path/to/your/config-template.yaml:/app/config.yaml --name OpenAOE opensealion/openaoe:latest
```

### Run by source code

#### **Install**

1. clone this project

```shell
git clone https://github.com/internlm/OpenAOE
```

2. \[_optional_\] build the frontend project when the frontend codes are changed

```shell
cd OpenAOE/openaoe/frontend
npm install
npm run build
```

#### **Start**

```shell
cd OpenAOE
pip install -r openaoe/backend/requirements.txt
python -m openaoe.main -f /path/to/your/config-template.yaml
```

> \[!TIP\]
> `/path/to/your/config-tempalte.yaml` is the configuration file loaded by OpenAOE at startup,
> which contains the relevant configuration information for the LLMs,
> including: API URLs, AKSKs, Tokens, etc.
> A template configuration yaml file can be found in `openaoe/backend/config/config-template.yaml`.
