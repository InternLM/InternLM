# InternLM Ecosystem

With the innovation waves driven by large language models (LLMs),  InternLM has been continuously building more comprehensive and powerful foundational models. It adheres to open-source and free commercial use, fully empowering the prosperity and development of the AI community ecosystem. It helps businesses and research institutions to lower the barriers to developing and applying LLMs, allowing the value of LLMs to shine in various industries.

The released InternLM supports a variety of well-known upstream and downstream projects, including LLaMA-Factory, vLLM, Langchain, and others, enabling a wide range of users to utilize the InternLM series models and open-source toolchains more efficiently and conveniently.

We categorize ecosystem projects into three main areas: Training, Inference, and Application. Each area features a selection of renowned open-source projects compatible with InternLM models. The list is continually expanding, and we warmly invite contributions from the community to include additional worthy projects.

## Training

### [InternEvo](https://github.com/InternLM/InternEvo)

InternEvo is an open-sourced lightweight training framework aims to support model pre-training without the need for extensive dependencies. It supports pre-training on large-scale clusters with thousands of GPUs

A quickstart guide for pre-training and fine-tuning the full series of InternLM models can be accessed from [here](https://github.com/InternLM/InternEvo/blob/develop/doc/en/usage.md)

### [XTuner](https://github.com/InternLM/xtuner)

XTuner is an efficient, flexible and full-featured toolkit for fine-tuning large models.

You can find the best practice for fine-tuning the InternLM series models in the [README](https://github.com/InternLM/InternLM/tree/main/finetune#xtuner)

### [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

LLaMA-Factory is an open-source, easy-to-use fine-tuning and training framework for LLMs

```bash
llamafactory-cli train \
    --model_name_or_path internlm/internlm2-chat-1_8b \
    --quantization_bit 4 --stage sft  --lora_target all \
    --dataset 'identity,alpaca_en_demo' --template intern2 \
    --output_dir output --do_train
```

### [swift](https://github.com/modelscope/swift)

SWIFT supports training, inference, evaluation and deployment of LLMs and MLLMs (multimodal large models).

```bash
swift sft --model_type internlm2-1_8b-chat \
    --model_id_or_path Shanghai_AI_Laboratory/internlm2-chat-1_8b  \
    --dataset AI-ModelScope/blossom-math-v2 --output_dir output
```

## Inference

### [LMDeploy](https://github.com/InternLM/lmdeploy)

LMDeploy is an efficient toolkit for compressing, deploying, and serving LLMs and VLMs.

With only 4 lines of code, you can perform `internlm2_5-7b-chat` inference after `pip install lmdeploy`:

```python
from lmdeploy import pipeline
pipe = pipeline("internlm/internlm2_5-7b-chat")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

### [vLLM](https://github.com/vllm-project/vllm)

`vLLM` is a high-throughput and memory-efficient inference and serving engine for LLMs.

After the installation via `pip install vllm`, you can conduct the `internlm2_5-7b-chat` model inference as follows:

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
llm = LLM(model="internlm/internlm2_5-7b-chat", trust_remote_code=True)
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

TGI is a toolkit for deploying and serving Large Language Models (LLMs). The easiest way of deploying a LLM is using the official Docker container:

```shell
model="internlm/internlm2_5-chat-7b"
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model
```

And then you can make requests like

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

### [llama.cpp](https://github.com/ggerganov/llama.cpp)

`llama.cpp` is a LLM inference framework developed in C/C++. Its goal is to enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware - locally and in the cloud.

`InternLM2` and `InternLM2.5` can be deployed with `llama.cpp` by following the below instructions:

- Refer [this](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#build) guide to build llama.cpp from source
- Convert the InternLM model to GGUF model and run it according to the [guide](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#prepare-and-quantize)

### [ollama](https://github.com/ollama/ollama)

Ollama bundles model weights, configuration, and data into a single package, defined by a Modelfile. It optimizes setup and configuration details, enabling users to easily set up and execute LLMs locally (in CPU and GPU modes).

The following snippet presents the Modefile of InternLM2.5 with `internlm2_5-7b-chat` as an example. Note that the model has to be converted to GGUF model at first.

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

Then, create an image from the above `Modelfile` like this:

```shell
ollama create internlm2.5:7b-chat -f ./Modelfile
```

Regarding the usage of `ollama`, please refer [here](https://github.com/ollama/ollama/tree/main/docs).

### [llamafile](https://github.com/Mozilla-Ocho/llamafile)

llamafile lets you turn large language model (LLM) weights into executables. It combines [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan).

The best practice of deploying InternLM2 or InternLM2.5 using llamafile is shown as below:

- Convert the model into GGUF model by `llama.cpp`. Suppose we get `internlm2_5-chat-7b.gguf` in this step
- Create the llamafile

```shell
wget https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.6/llamafile-0.8.6.zip
unzip llamafile-0.8.6.zip

cp llamafile-0.8.6/bin/llamafile internlm2_5.llamafile

echo "-m
internlm2_5-chat-7b.gguf
--host
0.0.0.0
-ngl
999
..." > .args

llamafile-0.8.6/bin/zipalign -j0 \
  internlm2_5.llamafile \
  internlm2_5-chat-7b.gguf \
  .args

rm -rf .args
```

- Run the llamafile

```shell
./internlm2_5.llamafile
```

Your browser should open automatically and display a chat interface. (If it doesn't, just open your browser and point it at http://localhost:8080)

### [mlx](https://github.com/ml-explore/mlx)

MLX is an array framework for machine learning research on Apple silicon, brought to you by Apple machine learning research.

With the following steps, you can perform InternLM2 or InternLM2.5 inference on Apple devices.

- Installation

```shell
pip install mlx mlx-lm
```

- Inference

```python
from mlx_lm import load, generate
tokenizer_config = {"trust_remote_code": True}
model, tokenizer = load("internlm/internlm2-chat-1_8b", tokenizer_config=tokenizer_config)
response = generate(model, tokenizer, prompt="write a story", verbose=True)
```

## Application

### [Langchain](https://github.com/langchain-ai/langchain)

LangChain is a framework for developing applications powered by large language models (LLMs).

You can build a [LLM chain](https://python.langchain.com/v0.1/docs/get_started/quickstart/#llm-chain) by the OpenAI API. And the server is recommended to be launched by LMDeploy, vLLM or others that are compatible with openai server.

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

Or you can follow the guide [here](https://python.langchain.com/v0.1/docs/get_started/quickstart/#llm-chain) and run an ollama model locally.

As for other user cases, please look for them from [here](https://python.langchain.com/v0.1/docs/get_started/introduction/).

### [LlamaIndex](https://github.com/run-llama/llama_index)

LlamaIndex is a framework for building context-augmented LLM applications.

It chooses ollama as the LLM inference engine locally. An example can be found from the [Starter Tutorial(Local Models)](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/).

Therefore, you can integrate InternLM2 or InternLM2.5 models to LlamaIndex smoothly if you can deploying them with `ollama` as guided in the [ollama section](#ollama)


### [LazyLLM](https://github.com/LazyAGI/LazyLLM)

LazyLLM is an framework which supports the easiest and laziest way for building multi-agent LLMs applications. It offers extremely high flexibility and ease of use compared to LangChain and LLamaIndex.

When you have installed `lazyllm` by `pip3 install lazyllm` and `lazyllm install standard`, you can use the following code to build chatbots based on internLM at a very low cost, without worrying about the special tokens (such as `<|im_start|>system` and `<|im_end|>`) of the dialogue model. Don’t worry about not having weight files; as long as you are connected to the internet, the code below will automatically download the weight files and deploy the service for you. Enjoy the convenience that LazyLLM brings to you.

```python
from lazyllm import TrainableModule, WebModule
# Model will be download automatically if you have an internet connection
m = TrainableModule('internlm2_5-7b-chat')
# will launch a chatbot server
WebModule(m).start().wait()
```

You can use the following code to finetune your model if needed. When the trainset (The dataset needs to be downloaded to the local machine, for example:[
alpaca_gpt4_zh](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh)) of the TrainableModule is set, during the calling of the WebModule's update function, the TrainableModule will be automatically fine-tuned, and then both the TrainableModule and the WebModule will be deployed separately.

```python
from lazyllm import TrainableModule, WebModule
m = TrainableModule('internlm2-chat-7b').trainset('/patt/to/your_data.json').mode('finetune')
WebModule(m).update().wait()
```

It is worth mentioning that regardless of which model in the InternLM series you use, you can perform inference and fine-tuning with LazyLLM. You don't need to worry about the model's segmentation strategy or special tokens.<br>
If you want to build your own RAG application, you don't need to first start the inference service and then configure the IP and port to launch the application like you would with LangChain. Refer to the code below, and with LazyLLM, you can use the internLM series models to build a highly customized RAG application in just ten lines of code, along with document management services (The document requires specifying the local absolute path. You can download it as an example from here: [rag_master](https://huggingface.co/datasets/Jing0o0Xin/rag_master)):

<details>
<summary>Click here to get imports and prompts</summary>

```python
import os
import lazyllm
from lazyllm import pipeline, parallel, bind, SentenceSplitter, Document, Retriever, Reranker

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, you need to provide your answer based on the given context and question.'
```
</details>

```python
documents = Document(dataset_path='/file/to/yourpath', embed=lazyllm.TrainableModule('bge-large-zh-v1.5'), create_ui=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)
    ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.TrainableModule("internlm2_5-7b-chat").prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))
lazyllm.WebModule(ppl, port=23456).start().wait()
```

LazyLLM Documents: https://docs.lazyllm.ai/
