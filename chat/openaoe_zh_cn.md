# OpenAOE 多模型对话

[English](openaoe.md) | 简体中文

## 介绍

[OpenAOE](https://github.com/InternLM/OpenAOE) 是一个 LLM-Group-Chat 框架，可以同时与多个商业大模型或开源大模型进行聊天。 OpenAOE还提供后端API和WEB-UI以满足不同的使用需求。

目前已经支持的大模型有：  [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat), [IntenLM-Chat-7B](https://huggingface.co/internlm/internlm-chat-7b), GPT-3.5, GPT-4, Google PaLM, MiniMax, Claude, 讯飞星火等。

## 快速安装

我们将提供 3 种不同的方式安装：基于 pip、基于 docker 以及基于源代码，实现开箱即用。

### 基于 pip

> \[!TIP\]
> 需要 python >= 3.9

#### **安装**

```shell
pip install -U openaoe
```

#### **运行**

```shell
openaoe -f /path/to/your/config-template.yaml
```

### 基于 docker

#### **安装**

有两种方式获取 OpenAOE 的 docker 镜像：

1. 官方拉取

```shell
docker pull opensealion/openaoe:latest
```

2. 本地构建

```shell
git clone https://github.com/internlm/OpenAOE
cd OpenAOE
docker build . -f docker/Dockerfile -t openaoe:latest
```

#### **运行**

```shell
docker run -p 10099:10099 -v /path/to/your/config-template.yaml:/app/config.yaml --name OpenAOE opensealion/openaoe:latest
```

### 基于源代码

#### **安装**

1. 克隆项目

```shell
git clone https://github.com/internlm/OpenAOE
```

2. \[_可选_\] （如果前端代码发生变动）重新构建前端项目

```shell
cd OpenAOE/openaoe/frontend
npm install
npm run build
```

#### **运行**

```shell
cd OpenAOE
pip install -r openaoe/backend/requirements.txt
python -m openaoe.main -f /path/to/your/config-template.yaml
```

> \[!TIP\]
> `/path/to/your/config-template.yaml` 是 OpenAOE 启动时读取的配置文件，里面包含了大模型的相关配置信息，
> 包括：调用API地址、AKSK、Token等信息，是 OpenAOE 启动的必备文件。模板文件可以在 `openaoe/backend/config/config-template.yaml` 中找到。
