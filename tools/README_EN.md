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

We can run the following command to generate `bin` and `meta` files for raw data, where the parameter `raw_data_name` indicates the file name of raw data, `input_file_type` denotes the raw data format, which should be `txt`, `json` and `jsonl`, and `bin` indicates the path to save the generated `bin` file.
```bash
$ python tools/tokenizer.py --raw_data_name your_raw_data_file_name(without suffix) --input_file_type 'text' or 'json' or 'jsonl' --bin your_output_bin_path
```

An example of data processing in `txt` format is given here (the data processing for `json` and `jsonl` is identical to that for `txt`).

Given a file `raw_data.txt` containg raw data with the following content.
```bash
Appreciate every detail in life to truly taste the flavor of happiness.
Dreams are the source of life’s motivation. Pursue them diligently to achieve your goals.
Learn to be tolerant and understanding to establish truly harmonious interpersonal relationships.
```
Next, we can run the following command to generate `bin` and `meta` files for raw data.
```bash
$ python tools/tokenizer.py --raw_data_name raw_data --input_file_type 'text' --bin cn/output.bin
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