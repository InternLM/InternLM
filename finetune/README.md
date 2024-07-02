# Fine-tuning with InternLM

English | [简体中文](./README_zh-CN.md)

We recommend two projects to fine-tune InternLM.

1. [XTuner](https://github.com/InternLM/xtuner) is an efficient, flexible and full-featured toolkit for fine-tuning large models.

2. [InternEvo](https://github.com/InternLM/InternEvo/) is a powerful training framework that supports large-scale pre-training and finetuning.

## XTuner

### Highlights

1. Support LLM, VLM pre-training / fine-tuning on almost all GPUs. XTuner is capable of fine-tuning InternLM2-7B on a single 8GB GPU, as well as multi-node fine-tuning of models exceeding 70B.
2. Support various training algorithms ([QLoRA](http://arxiv.org/abs/2305.14314), [LoRA](http://arxiv.org/abs/2106.09685), full-parameter fune-tune), allowing users to choose the most suitable solution for their requirements.
3. Compatible with [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀, easily utilizing a variety of ZeRO optimization techniques.
4. The output models can seamlessly integrate with deployment and server toolkit ([LMDeploy](https://github.com/InternLM/lmdeploy)), and large-scale evaluation toolkit ([OpenCompass](https://github.com/open-compass/opencompass), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)).

### Installation

- It is recommended to build a Python 3.10 virtual environment using conda

  ```bash
  conda create --name xtuner-env python=3.10 -y
  conda activate xtuner-env
  ```

- Install XTuner with DeepSpeed integration

  ```shell
  pip install -U 'xtuner[deepspeed]>=0.1.22'
  ```

### Fine-tune

XTuner supports the efficient fine-tune (*e.g.*, QLoRA) for InternLM2.

- **Step 0**, prepare the config. XTuner provides many ready-to-use configs and we can view all configs of InternLM2 by

  ```shell
  xtuner list-cfg -p internlm2
  ```

  Or, if the provided configs cannot meet the requirements, please copy the provided config to the specified directory and make specific modifications by

  ```shell
  xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
  vi ${SAVE_PATH}/${CONFIG_NAME}_copy.py
  ```

- **Step 1**, start fine-tuning.

  ```shell
  xtuner train ${CONFIG_NAME_OR_PATH}
  ```

  For example, we can start the QLoRA fine-tuning of InternLM2.5-Chat-7B with oasst1 dataset by

  ```shell
  # On a single GPU
  xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  # On multiple GPUs
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  (SLURM) srun ${SRUN_ARGS} xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2
  ```

  - `--deepspeed` means using [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 to optimize the training. XTuner comes with several integrated strategies including ZeRO-1, ZeRO-2, and ZeRO-3. If you wish to disable this feature, simply remove this argument.

- **Step 2**, convert the saved PTH model (if using DeepSpeed, it will be a directory) to HuggingFace model, by

  ```shell
  xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
  ```

### Chat

XTuner provides tools to chat with pretrained / fine-tuned large models.

```shell
xtuner chat ${NAME_OR_PATH_TO_LLM} [optional arguments]
```

For example, we can start the chat with InternLM2.5-Chat-7B :

```shell
xtuner chat internlm/internlm2_5-chat-7b --prompt-template internlm2_chat
```

## InternEvo

\[TODO\]
