This directory provide some tools for model training with the following file structure.

```bash
├── transformers  # tools for adapting Hugging Face's transformers
│   ├── configuration_internlm.py  # tools for adapting config
│   ├── modeling_internlm.py  # tools for adapting model
│   └── tokenization_internlm.py  # tools for adapting tokenizer
│   └── convert2hf.py  # tools for adapting models to Hugging Face's format
└── tokenizer.py  # tools for generating `bin` and `meta` file for raw data
```

# tokenizer.py

We need to use a `tokenizer` to generate `bin` and `meta` files for raw data. We import the tokenizer model by specifying the model weight path in `tools/tokenizer.py`. Currently, we provide `V7.model` to generate tokens. If you want to use a different model, you can modify the model weight path in `tokenizer.py` directly.

We can run the following command to generate `bin` and `meta` files corresponding to the original data. The parameter `text_input_path` represents the path of the original text data, currently supporting `txt`, `json`, and `jsonl` formats, while `bin_output_path` represents the save path of the generated `bin` files.
```bash
$ python tools/tokenizer.py --text_input_path your_input_text_path --bin_output_path your_output_bin_path
```

An example of data processing in `txt` format is given here:

Given a file `raw_data.txt` containg raw data with the following content.

```bash
Appreciate every detail in life to truly taste the flavor of happiness.
Dreams are the source of life’s motivation. Pursue them diligently to achieve your goals.
Learn to be tolerant and understanding to establish truly harmonious interpersonal relationships.
```

Next, we can run the following command to generate `bin` and `meta` files for raw data.

```bash
$ python tools/tokenizer.py --text_input_path your_input_text_path --bin_output_path your_output_bin_path
```

It should be noted that the generated `bin` files should be placed in one of the following directories to clarify the data type: `cn`(Chinese), `en`(English), `code`(code data), `ja`(Japanese), `ar`(Arabic) and `kaoshi`(kaoshi data).

The format of generated `bin` file is as follows.

```python
{"tokens": [98655, 2317, 2922, 6649, 1595, 7856, 435, 2424, 442, 9556, 12807, 410, 17313, 446, 23331, 95746]}
{"tokens": [98655, 302, 1383, 269, 657, 410, 2687, 446, 2424, 98667, 269, 25220, 281, 523, 1874, 492, 1248, 38127, 4563, 442, 11227, 829, 8980, 95746]}
{"tokens": [98655, 24190, 442, 517, 15013, 649, 454, 8793, 442, 5849, 9556, 17917, 1369, 1084, 29890, 12021, 95746]}
```

In the generated `bin` file, each line (`sequence`) corresponds to the `tokens` for each sentence in the raw data.

The format of generated `meta` file in as follows.

```bash
(0, 16), (110, 24), (262, 17)
```

Each tuple in the `meta` file represents the meta information of each `sequence` where the first element in the tuple indicates the `starting index` of each `sequence` among all `sequences` and the second element indicates the amount of `tokens` for each `sequence`.

For example, the `starting index` is 0 for the first `sequence` with 16 `tokens`. Since the length of `sequence` in `string` format is 109, the `starting index` is 110. And the number of `tokens` of the sencond `sequence` is 24.

The `bin` and `meta` file formats for `json` and `jsonl` type files are the same as for `txt`, so we won't go over them here.

# pal_inference.py

Perform reasoning using [PAL](https://github.com/reasoning-machines/pal) on the [GSM8K](https://huggingface.co/datasets/gsm8k) dataset, allowing the model to generate code and solve mathematical problems through Python interpretation. Here's how you can use it:

```bash
# Usage:
python pal_inference.py <model> <out_dir> [--dataset <dataset>] [--max_length <length>] [--top_p <threshold>] [--eoh <end token>] [--eoa <end token>] [--eos <end token>] [--temperature <temp>] [--time_out <time>] [--verbose, -v] [--append, -a]

# Parameters:
# <model>                   Path to the model used for inference.
# <out_dir>                 Generated code will be saved in the specified output folder.

# Optional arguments:
# --dataset <dataset>       Dataset name used for code generation (default: gsm8k).
# --max_length <length>     Model's maximum input token length (default: 2048).
# --top_p <threshold>       Probability threshold for candidate tokens (default: 0.8).
# --eoh <end token>         End of human (user) token. (default: "").
# --eoa <end token>         End of assistant (bot) token. (default: "").
# --eos <end token>         End of system token. (default: "").
# --temperature, -t <temp>  Sampling temperature during generation (default: 1.0).
# --time_out <time>         Maximum time (in seconds) for executing the generated code (default: 100).
# --verbose, -v             Print code error messages (optional).
# --append, -a              ppend the output to historical results (optional).
```

Below is an example of usage:

```bash
python tools/pal_inference.py internlm/internlm-chat-7k ./output -v
```

The output file contains each line with the input question, the correct answer, the executed answer, the score, and the Python code block generated by the model:

````json
{
    "question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "target": 18.0,
    "answer": 18.0,
    "score": 1,
    "generation": ["```python\ndef solution():\n    eggs_per_day = 16\n    eggs_per_breakfast = 3\n    eggs_per_muffin = 4\n    eggs_used = eggs_per_day - eggs_per_breakfast - eggs_per_muffin\n    eggs_sold = eggs_used\n    price_per_egg = 2\n    eggs_made = eggs_sold * price_per_egg\n    result = eggs_made\n    return result\n```"]
}
````

InternLM performance in the GSM8K dataset with and without tools:

| Method   | **InternLM-Chat-7B** |
| -------- | -------------------- |
| w/o tool | 34.5                 |
| w tool   | 39.2                 |
