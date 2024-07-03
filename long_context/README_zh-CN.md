# InternLM with Long Context

## InternLM2.5-7B-Chat-1M

很高兴向大家介绍 InternLM2.5-7B-Chat-1M，它拥有处理超长文本的能力，支持长达 1M tokens 的输入。

在预训练阶段，我们使用了包含长度为 256K tokens 的语料训练。为了应对由于数据同质可能引起的领域偏移问题，我们在训练过程中引入了合成数据，不仅保持了模型的能力，还增强了其对上下文的理解程度。

在“*大海捞针*”实验中，InternLM2.5-7B-Chat-1M 能够在长达 1M tokens 的文档中准确地定位关键信息。

<p align="center">
<img src="https://github.com/libowen2121/InternLM/assets/19970308/2ce3745f-26f5-4a39-bdcd-2075790d7b1d" alt="drawing" width="700"/>
</p>

同时，我们还采用了 [LongBench](https://github.com/THUDM/LongBench) 基准来评估长文档理解能力。InternLM2.5-7B-Chat-1M 在测试中相较于同类型的模型达到了最佳性能。

<p align="center">
<img src="https://github.com/libowen2121/InternLM/assets/19970308/1e8f7da8-8193-4def-8b06-0550bab6a12f" alt="drawing" width="800"/>
</p>

## 使用 InternLM2.5-1M 进行文档聊天

下面介绍如何使用 InternLM2.5-7B-Chat-1M 来根据输入文档进行聊天。为了获得最佳性能，尤其是在处理长文本输入时，我们推荐使用 [LMDeploy](https://github.com/InternLM/LMDeploy) 来进行模型部署。

### 支持的文件类型

当前版本支持 PDF、TXT 和 Markdown 三类文件。未来我们将很快支持更多文件类型！

- TXT 和 Markdown 文件：直接读取，无需转换。
- PDF 文件：为了高效处理 PDF 文件，我们推出了轻量级的开源工具 [Magic-Doc](https://github.com/magicpdf/Magic-Doc) ，其可以将多种文件类型转换为 Markdown 格式。

### 安装

开始前，请安装所需的依赖：

```bash
pip install "fairy-doc[cpu]"
pip install streamlit
pip install lmdeploy
```

### 部署模型

从 [model zoo](../README.md#model-zoo) 下载模型。

通过以下命令部署模型。用户可以指定 `session-len`（sequence length）和 `server-port` 来定制模型推理。

```bash
lmdeploy serve api_server {path_to_hf_model} \
--model-name internlm2-chat \
--session-len 65536 \
--server-port 8000
```

要进一步增加序列长度，建议添加以下参数：
`--max-batch-size 1 --cache-max-entry-count 0.7 --tp {num_of_gpus}`

### 启动 Streamlit demo

```bash
streamlit run long_context/doc_chat_demo.py \
-- --base_url http://0.0.0.0:8000/v1
```

用户可以根据需要指定端口。如果在本地运行 demo，URL 可以是 `http://0.0.0.0:{your_port}/v1` 或 `http://localhost:{your_port}/v1`。对于云服务器，我们推荐使用 VSCode 来启动 demo，以实现无缝端口转发。

对于长输入，我们建议使用以下参数：

- Temperature: 0.05
- Repetition penalty: 1.02

当然，用户也可以根据需要在 web UI 中调整这些参数以获得最佳效果。

下面是效果演示视频：

https://github.com/libowen2121/InternLM/assets/19970308/1d7f9b87-d458-4f24-9f7a-437a4da3fa6e

## 🔜 敬请期待更多

我们将不断优化和更新长文本模型，以提升其在长文本上的理解和分析能力。敬请关注！
