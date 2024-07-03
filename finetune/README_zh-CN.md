# 微调 InternLM

[English](./README.md) | 简体中文

我们推荐以下两种框架微调 InternLM：

1. [XTuner](https://github.com/InternLM/xtuner) 是一个高效、灵活、全能的轻量化大模型微调工具库。

2. [InternEvo](https://github.com/InternLM/InternEvo/) 是一个支持大规模预训练和微调的训练框架。

## XTuner

### 亮点

1. 支持大语言模型 LLM、多模态图文模型 VLM 的预训练及轻量级微调。XTuner 支持在 8GB 显存下微调 7B 模型，同时也支持多节点跨设备微调更大尺度模型（70B+）。
2. 支持 [QLoRA](http://arxiv.org/abs/2305.14314)、[LoRA](http://arxiv.org/abs/2106.09685)、全量参数微调等多种微调算法，支撑用户根据具体需求作出最优选择。
3. 兼容 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀，轻松应用各种 ZeRO 训练优化策略。
4. 训练所得模型可无缝接入部署工具库 [LMDeploy](https://github.com/InternLM/lmdeploy)、大规模评测工具库 [OpenCompass](https://github.com/open-compass/opencompass) 及 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)。

### 安装

- 借助 conda 准备虚拟环境

  ```bash
  conda create --name xtuner-env python=3.10 -y
  conda activate xtuner-env
  ```

- 安装集成 DeepSpeed 版本的 XTuner

  ```shell
  pip install -U 'xtuner[deepspeed]>=0.1.22'
  ```

### 微调

- **步骤 0**，准备配置文件。XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看所有 InternLM2 的预置配置文件：

  ```shell
  xtuner list-cfg -p internlm2
  ```

  或者，如果所提供的配置文件不能满足使用需求，请导出所提供的配置文件并进行相应更改：

  ```shell
  xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
  vi ${SAVE_PATH}/${CONFIG_NAME}_copy.py
  ```

- **步骤 1**，开始微调。

  ```shell
  xtuner train ${CONFIG_NAME_OR_PATH}
  ```

  例如，我们可以利用 QLoRA 算法在 oasst1 数据集上微调 InternLM2.5-Chat-7B：

  ```shell
  # 单卡
  xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  # 多卡
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  (SLURM) srun ${SRUN_ARGS} xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2
  ```

  - `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

- **步骤 2**，将保存的 PTH 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace 模型：

  ```shell
  xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
  ```

### 对话

XTuner 提供与大模型对话的工具。

```shell
xtuner chat ${NAME_OR_PATH_TO_LLM} [optional arguments]
```

例如：

与 InternLM2.5-Chat-7B 对话：

```shell
xtuner chat internlm/internlm2_5-chat-7b --prompt-template internlm2_chat
```

## InternEvo

\[TODO\]
