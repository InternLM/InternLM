import argparse
import json
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "V7_sft.model")
sys.path.append(os.path.join(current_dir, "transformers"))
from tokenization_internlm import InternLMTokenizer

tokenizer = InternLMTokenizer(vocab_file=model_path)


def write_bin(context: str, bin_file) -> None:
    """
    Write bin file based on the context.

    Args:
        context (str): the context of raw file.
        bin_file (file handler): the opened bin file.

    Example:
    >>> write_bin("今天天气晴朗适合出门散步", "out.bin") # the output file format is 'txt'
    >>> out.bin
    >>> {"tokens": [67577, 69095, 63010, 61770, 67783, 69301, 74732]}
    """
    # encode the context into tokens, which is a list, eg. [67577, 69095, 63010, 61770, 67783, 69301, 74732]
    tokens = tokenizer.encode(context)
    # transfer the list into dic, key is str 'tokens', value is tokens.
    # eg. {"tokens": [67577, 69095, 63010, 61770, 67783, 69301, 74732]}
    data = dict(tokens=tokens)
    # encode the data into bytes to save
    saved_bin = str.encode(json.dumps(data) + "\n")

    # write bytes into bin_file
    bin_file.write(saved_bin)


def prepare_meta(bin_output_path: str):
    """
    Prepare metadata for the given bin file.

    Args:
        bin_output_path (str): Output bin file path.
    """
    meta = []
    cur = 0
    with open(bin_output_path, "rb") as f:
        while True:
            # read lines
            line = f.readline()
            # if line is empty, then break
            if line == b"":
                break
            # obtain the token amount of each line
            length = len(json.loads(line)["tokens"])
            # meta is a list of tuple(cur, length)
            # cur: the start index of each line
            # length: the token amount of each line
            meta.append((cur, length))
            # update the cur to generate the meta information of next line
            cur += len(line)

    # define path of the generated meta file
    meta_fp = bin_output_path + ".meta"
    # save the generated meta information
    with open(meta_fp, "wb") as f:
        meta = np.array(meta, dtype=np.int32)
        np.save(f, meta)


def text2bin(text_input_path: str, bin_output_path: str):
    """
    Read content from the input file and write to bin file.
    Currently support 3 input formats: 'txt', 'json' and 'jsonl'.

    Args:
        text_input_path (str): txt file path.
        bin_output_path (str): output bin file path.
    """
    # Check if the txt file exists
    if not os.path.isfile(text_input_path):
        raise FileNotFoundError(f"{text_input_path} does not exist.")

    file_format = text_input_path.split(".")[-1]
    assert file_format in ["txt", "json", "jsonl"], print(
        "Invalid input file type. Currently support `txt`, `json` and `jsonl`."
    )

    with open(text_input_path, "r") as text_file, open(bin_output_path, "ab") as bin_file:
        if file_format == "txt":
            for line in text_file:
                # Strip any leading/trailing whitespace
                stripped_line = line.strip()
                if stripped_line:
                    # Pass each line to the write_bin function
                    write_bin(stripped_line, bin_file)

        elif file_format == "json":
            data = json.load(text_file)
            # assuming data is a list of dictionaries
            for record in data:
                # the type of record is dict, transfer the dict into str
                context = json.dumps(record)
                # encode the str and write into bin
                write_bin(context, bin_file)

        elif file_format == "jsonl":
            for line in text_file:
                # encode the str and write into bin
                write_bin(line, bin_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_input_path",
        type=str,
        required=True,
        help="Path to the input text file.",
    )
    parser.add_argument("--bin_output_path", type=str, required=True, help="Path to the output bin file.")

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    text2bin(args.text_input_path, args.bin_output_path)
    print(f"Successfully converted {args.text_input_path} to {args.bin_output_path}")

    # To avoid potential read/write errors, the metadata preparation follows after creating the .bin file.
    prepare_meta(args.bin_output_path)
    print(f"Successfully generated {args.bin_output_path}.meta")


if __name__ == "__main__":
    main()
