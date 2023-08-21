import argparse
import json
import os.path as osp
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def process(dataset_path, sp_model):
    """Process data sample from input dataset

    Args:
        dataset_path (str): Path of dataset json file.
        sp_model (str): Path of tokenizer.

    Yields:
        tuple: dumped processed data sample and length of tokens.
    """

    dataset = json.load(open(dataset_path))

    for data in dataset:
        yield tokenize(get_chat_format_data(data), sp_model)


def get_chat_format_data(ori_data):
    """Format original data

    Args:
        ori_data (dict): input data sample.

    Returns:
        dict: data sample with chat format.
    """
    input_str = ori_data["input"]
    instruction_str = ori_data["instruction"]
    output_str = ori_data["output"]
    data = dict()
    if input_str != "":
        data["user"] = f"<|User|>:{instruction_str}\n{input_str}"
    else:
        data["user"] = f"<|User|>:{instruction_str}"
    data["bot"] = f"<|Bot|>:{output_str}"
    return data


def tokenize(sample, sp_model):
    """Tokenize input dataset

    Args:
        sample (dict): Input data sample.
        sp_model (str): Path of tokenizer.

    Returns:
        tuple: dumped processed data sample and length of tokens.
    """
    special_tokens_map = {"<eoh>": 103167, "<eoa>": 103166, "nl_id": 13}
    token_ids = [sp_model.bos_id()]
    human_s = sample["user"]
    ass_s = sample["bot"]

    human_ids = sp_model.encode(human_s) + [special_tokens_map["<eoh>"], special_tokens_map["nl_id"]]
    human_ids_ignore = [-token_id for token_id in human_ids]

    ass_template_ids = sp_model.encode("<|Bot|>:")
    ass_template_ids_ignore = [-token_ids for token_ids in ass_template_ids]
    ass_ids = (
        ass_template_ids_ignore
        + sp_model.encode(ass_s[8:])
        + [special_tokens_map["<eoa>"], special_tokens_map["nl_id"]]
    )

    token_ids += human_ids_ignore + ass_ids
    if len(token_ids) > 2047:
        token_ids = token_ids[:2047]
    token_ids += [sp_model.eos_id()]
    line = str.encode(json.dumps({"tokens": token_ids}) + "\n")
    return line, len(token_ids)


def dump_bin_meta_bin(samples, path, split_ratio=0.1):
    """Dump processed dataset

    Args:
        samples (dict): Input data sample.
        path (str): Path for output dataset.
        split_ratio (float): Ratio for validation dataset splitting.
            Default to: 0.1.

    Returns:
        tuple: number of train/valid tokens of processed dataset,
            number of train/valid samples of processed dataset.
    """

    train_path = osp.join(path, "train/en/")
    valid_path = osp.join(path, "valid/en/")
    train_dir = Path(train_path)
    valid_dir = Path(valid_path)
    train_dir.mkdir(exist_ok=True, parents=True)
    valid_dir.mkdir(exist_ok=True, parents=True)
    train_f = open(train_dir.joinpath("dataset.bin"), "wb")
    valid_f = open(valid_dir.joinpath("dataset.bin"), "wb")

    train_tokens = 0
    valid_tokens = 0
    last_train_position = 0
    last_valid_position = 0
    train_samples = 0
    valid_samples = 0
    train_meta = []
    valid_meta = []

    sample_length = len(samples)
    np.random.seed(0)
    valid_indices = np.random.choice(range(sample_length), int(sample_length * split_ratio)).tolist()

    count = -1
    for line, token_num in samples:
        count += 1
        if count in valid_indices:
            valid_tokens += token_num
            valid_f.write(line)
            valid_meta.append((last_valid_position, token_num))
            last_valid_position += len(line)
            valid_samples += 1
        else:
            train_tokens += token_num
            train_f.write(line)
            train_meta.append((last_train_position, token_num))
            last_train_position += len(line)
            train_samples += 1

    train_f.close()
    valid_f.close()
    np.save(open(train_dir.joinpath("dataset.bin.meta"), "wb"), train_meta)
    np.save(open(valid_dir.joinpath("dataset.bin.meta"), "wb"), valid_meta)

    return train_tokens, valid_tokens, train_samples, valid_samples


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="path of dataset json file")
    parser.add_argument("output_path", type=str, help="path of processed dataset")
    parser.add_argument("tokenizer_path", type=str, help="path of tokenizer")
    parser.add_argument("--split_ratio", type=float, default=0.1, help="ratio for validation dataset splitting")

    args = parser.parse_args()
    sp_model = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    split_ratio = args.split_ratio
    samples = []

    dataset = process(args.dataset_path, sp_model)
    for sample in tqdm(dataset):
        samples.append(sample)

    train_tokens, valid_tokens, train_samples, valid_samples = dump_bin_meta_bin(
        samples, args.output_path, args.split_ratio
    )
    print(f"number of train dataset: {train_samples}, number of train dataset token: {train_tokens}")
    print(f"number of validation dataset: {valid_samples}, number of validation dataset token: {valid_tokens}")
