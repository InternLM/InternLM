# InternLM with Long Context

English | [简体中文](./README_zh-CN.md)

## InternLM2.5 with 1M Context Length

We introduce InternLM2.5-7B-Chat-1M, a model developed to support extensive long inputs of up to 1M tokens.
This enhancement significantly enhances the model's ability to handle ultra-long text applications. See [model zoo](../README.md#model-zoo) for download or [model cards](../model_cards/) for more details.

During pre-training, we utilized natural language corpora with text lengths of 256K tokens. To address the potential domain shift caused by homogeneous data, we supplemented with synthetic data to maintain the model's capabilities while expanding its context.

We employed the "*needle in a haystack approach*" to evaluate the model's ability to retrieve information from long texts. Results show that InternLM2.5-7B-Chat-1M can accurately locate key information in documents up to 1M tokens in length.

<p align="center">
<img src="https://github.com/libowen2121/InternLM/assets/19970308/2ce3745f-26f5-4a39-bdcd-2075790d7b1d" alt="drawing" width="700"/>
</p>

We also used the [LongBench](https://github.com/THUDM/LongBench) benchmark to assess long-document comprehension capabilities. Our model achieved optimal performance in these tests.

<p align="center">
<img src="https://github.com/libowen2121/InternLM/assets/19970308/1e8f7da8-8193-4def-8b06-0550bab6a12f" alt="drawing" width="800"/>
</p>

## File Chat with InternLM2.5-1M

This section provides a brief overview of how to chat with InternLM2.5-7B-Chat-1M using an input document. For optimal performance, especially with extensively long inputs, we highly recommend using [LMDeploy](https://github.com/InternLM/LMDeploy) for model serving.

### Supported Document Types

Currently, we support PDF, TXT, and Markdown files, with more file types to be supported soon!

- TXT and Markdown files: These can be processed directly without any conversions.
- PDF files: We have developed [Magic-Doc](https://github.com/magicpdf/Magic-Doc), a lightweight open-source tool, to convert multiple file types to Markdown.

### Installation

To get started, install the required packages:

```bash
pip install "fairy-doc[cpu]"
pip install streamlit
pip install lmdeploy
```

### Deploy the Model

Download our model from [model zoo](../README.md#model-zoo).

Deploy the model using the following command. You can specify the `session-len` (sequence length) and `server-port`.

```bash
lmdeploy serve api_server {path_to_hf_model} \
--model-name internlm2-chat \
--session-len 65536 \
--server-port 8000
```

To further enlarge the sequence length, we suggest adding the following arguments:
`--max-batch-size 1 --cache-max-entry-count 0.7 --tp {num_of_gpus}`

### Launch the Streamlit Demo

```bash
streamlit run long_context/doc_chat_demo.py \
-- --base_url http://0.0.0.0:8000/v1
```

You can specify the port as needed. If running the demo locally, the URL could be `http://0.0.0.0:{your_port}/v1` or `http://localhost:{your_port}/v1`. For virtual cloud machines, we recommend using VSCode for seamless port forwarding.

For long inputs, we suggest the following parameters:

- Temperature: 0.05
- Repetition penalty: 1.02

Of course, you can tweak these settings for optimal performance yourself in the web UI.

The effect is demonstrated in the video below.

https://github.com/libowen2121/InternLM/assets/19970308/1d7f9b87-d458-4f24-9f7a-437a4da3fa6e

## 🔜 Stay Tuned for More

We are continuously enhancing our models to better understand and reason with extensive long inputs. Expect new features, improved performance, and expanded capabilities in upcoming updates!
