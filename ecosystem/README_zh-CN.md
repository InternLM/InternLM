# InternLM 生态

面向大模型掀起的新一轮创新浪潮，书生浦语（InternLM）持续打造综合能力更强大的基础模型，并坚持通过开源开放、免费商用，全面赋能整个AI社区生态的繁荣发展，帮助企业和研究机构降低大模型的开发和应用门槛，让大模型的价值在各行各业中绽放。

已发布的 InternLM 全系列模型，支持包括 LLaMA-Factory、vLLM、Langchain 等众多知名上下游项目。广大用户可以更高效、便捷的使用书生浦语系列模型与开源工具链。

我们将生态系统项目分为三个主要领域：训练、推理和应用。每个领域会展示了一些与 InternLM 模型兼容的著名开源项目。这个列表在不断扩展，我们热情邀请社区贡献，包括更多有价值的项目。

## 训练

### [InternEvo](https://github.com/InternLM/InternEvo)

InternEvo 是一个开源的轻量级训练框架，旨在支持无需大量依赖关系的模型预训练。凭借单一代码库，InternEvo 支持在具有上千 GPU 的大规模集群上进行预训练。

InternLM 全系列模型预训练和微调的快速入门指南可以查看[这里](https://github.com/InternLM/InternEvo/blob/develop/doc/en/usage.md)。

### [XTuner](https://github.com/InternLM/xtuner)

XTuner 是一个高效、灵活、全能的轻量化大模型微调工具库。

你可以在 [README](https://github.com/InternLM/InternLM/tree/main/finetune#xtuner) 中找到 InternLM 全系列模型微调的最佳实践。

### [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

LLaMA-Factory 是一个开源的、易于使用的 LLMs 微调和训练框架。

```bash
llamafactory-cli train \
    --model_name_or_path internlm/internlm2-chat-1_8b \
    --quantization_bit 4 --stage sft  --lora_target all \
    --dataset 'identity,alpaca_en_demo' --template intern2 \
    --output_dir output --do_train
```

### [swift](https://github.com/modelscope/swift)

```bash
swift sft --model_type internlm2-1_8b-chat \
    --model_id_or_path Shanghai_AI_Laboratory/internlm2-chat-1_8b  \
    --dataset AI-ModelScope/blossom-math-v2 --output_dir output
```

SWIFT 支持 LLMs 和多模态大型模型（MLLMs）的训练、推理、评估和部署。

## 推理

### [LMDeploy](https://github.com/InternLM/lmdeploy)

LMDeploy 是一个高效且友好的 LLMs 模型部署工具箱，功能涵盖了量化、推理和服务。

通过 `pip install lmdeploy` 安装后，只用以下 4 行代码，即可使用 `internlm2_5-7b-chat` 模型完成 prompts 的批处理：

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2_5-7b-chat")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

### [vLLM](https://github.com/vllm-project/vllm)

vLLM 是一个用于 LLMs 的高吞吐量和内存效率的推理和服务引擎。

通过 `pip install vllm` 安装后，你可以按照以下方式使用 `internlm2_5-chat-7b` 模型进行推理：

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="internlm/internlm2_5-chat-7b", trust_remote_code=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### [TGI](https://github.com/huggingface/text-generation-inference)

TGI 是一个用于部署和提供 LLMs 服务的工具包。部署 LLM 服务最简单的方法是使用官方的 Docker 容器：

```shell
model="internlm/internlm2_5-chat-7b"
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model
```

然后，可以采用下述方式发送请求：

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

### [llama.cpp](https://github.com/ggerganov/llama.cpp)

llama.cpp 是一个用 C/C++ 开发的 LLMs 推理框架。其目标是在各种硬件上实现最小设置和最先进的性能的 LLM 推理——无论是在本地还是在云端。

通过以下方式可以使用 llama.cpp 部署 InternLM2 和 InternLM2.5 模型：

- 参考 [这里](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#build) 编译并安装 llama.cpp
- 把 InternLM 模型转成 GGUF 格式，具体方法参考 [此处](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#prepare-and-quantize)

### [ollama](https://github.com/ollama/ollama)

Ollama 将模型权重、配置和数据打包到一个单一的包中，由 Modelfile 定义。它优化了安装和配置，使用户能够轻松地在本地（以 CPU 和 GPU 模式）设置和执行 LLMs。

以下展示的是 `internlm2_5-7b-chat` 的 Modelfile。请注意，应首先把模型转换为 GGUF 模型。

```shell
echo 'FROM ./internlm2_5-7b-chat.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<im_end>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER stop "<|action_end|>"
PARAMETER stop "<|im_end|>"

SYSTEM """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
' > ./Modelfile
```

接着，使用上述 `Modelfile` 创建镜像：

```shell
ollama create internlm2.5:7b-chat -f ./Modelfile
```

Ollama 的使用方法可以参考[这里](https://github.com/ollama/ollama/tree/main/docs)。

### [llamafile](https://github.com/Mozilla-Ocho/llamafile)

llamafile 可以把 LLMs 的权重转换为可执行文件。它结合了 llama.cpp 和 Cosmopolitan Libc。

使用 llamafile 部署 InternLM 系列模型的最佳实践如下：

- 通过 llama.cpp 将模型转换为 GGUF 模型。假设我们在这一步得到了 `internlm2_5-chat-7b.gguf`
- 创建 llamafile

```shell
wget https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.6/llamafile-0.8.6.zip
unzip llamafile-0.8.6.zip

cp llamafile-0.8.6/bin/llamafile internlm2_5.llamafile

echo "-m
internlm2_5-7b-chat.gguf
--host
0.0.0.0
-ngl
999
..." > .args

llamafile-0.8.6/bin/zipalign -j0 \
  internlm2_5.llamafile \
  internlm2_5-7b-chat.gguf \
  .args

rm -rf .args
```

- Run the llamafile

```shell
./internlm2_5.llamafile
```

你的浏览器应该会自动打开并显示一个聊天界面。（如果没有，只需打开你的浏览器并访问 http://localhost:8080）

### [mlx](https://github.com/ml-explore/mlx)

MLX 是苹果公司为用户在苹果芯片上进行机器学习提供的一套框架。

通过以下步骤，你可以在苹果设备上进行 InternLM2 或者 InternLM2.5 的推理。

- 安装

```shell
pip install mlx mlx-lm
```

- 推理

```python
from mlx_lm import load, generate
tokenizer_config = {"trust_remote_code": True}
model, tokenizer = load("internlm/internlm2-chat-1_8b", tokenizer_config=tokenizer_config)
response = generate(model, tokenizer, prompt="write a story", verbose=True)
```

## 应用

### [Langchain](https://github.com/langchain-ai/langchain)

LangChain 是一个用于开发由 LLMs 驱动的应用程序的框架。

你可以通过 OpenAI API 构建一个 [LLM 链](https://python.langchain.com/v0.1/docs/get_started/quickstart/#llm-chain)。建议使用 LMDeploy、vLLM 或其他与 OpenAI 服务兼容的部署框架来启动服务。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    api_key="a dummy key",
    base_ur='https://0.0.0.0:23333/v1')
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm

chain.invoke({"input": "how can langsmith help with testing?"})
```

或者，你可以按照[这份指南](https://python.langchain.com/v0.1/docs/get_started/quickstart/#llm-chain)在本地使用 ollama 推理浦语模型。

对于其他使用方式，请从[这里](https://python.langchain.com/v0.1/docs/get_started/introduction/)查找。

### [LlamaIndex](https://github.com/run-llama/llama_index)

LlamaIndex 是一个用于构建上下文增强型 LLM 应用程序的框架。

它选择 ollama 作为 LLM 推理引擎。你可以在[入门教程（本地模型）](<(https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)>)中找到示例。

因此，如果能够按照 [ollama 章节](#ollama)使用 ollama 部署浦语模型，你就可以顺利地将浦语模型集成到 LlamaIndex 中。
