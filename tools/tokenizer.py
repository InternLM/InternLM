import argparse
import json
import os
import warnings

import numpy as np
from sentencepiece import SentencePieceProcessor
from termcolor import colored

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "V7.model")
tokenizer = SentencePieceProcessor(model_file=model_path)


def write_bin(context: str, path: str) -> None:
    """
    Write bin file.

    Args:
        context (str): the context of raw file.
        path (str): the path for output bin file.

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

    # write bytes into bin path
    with open(path, "ab") as f:
        f.write(saved_bin)


def prepare_meta(bin_file_path: str):
    """
    Prepare metadata for the given bin file.

    Args:
        bin_file_path (str): the bin file path.
    """
    meta = []
    cur = 0
    with open(bin_file_path, "rb") as f:
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
    print(meta)
    # define path of the generated meta file
    meta_fp = bin_file_path + ".meta"
    # save the generated meta information
    with open(meta_fp, "wb") as f:
        meta = np.array(meta, dtype=np.int32)
        np.save(f, meta)


def txt2bin(txt_file_path: str, bin_file_path: str):
    """
    Read content from txt file and write to bin file

    Args:
        txt_file_path (str): txt file path.
        bin_file_path (str): output bin file path.
    """
    # Check if the txt file exists
    if not os.path.isfile(txt_file_path):
        warnings.warn(colored(f"{txt_file_path} does not exist.", "red"))
        return

    try:
        # Open the text file
        with open(txt_file_path, "r") as txt_file:
            for line in txt_file:
                # Strip any leading/trailing whitespace
                stripped_line = line.strip()
                if stripped_line:
                    # Pass each line to the write_bin function
                    write_bin(stripped_line, bin_file_path)

        print(colored(f"Successfully converted {txt_file_path} to {bin_file_path}", "green"))

    except Exception as e:
        print(colored(f"Error while converting {txt_file_path} to {bin_file_path}: {str(e)}", "red"))


def json2bin(json_file_path: str, bin_file_path: str):
    """
    Read content from json file and write to bin file

    Args:
        json_file_path (str): json file path.
        bin_file_path (str): output bin file path.
    """

    if not os.path.isfile(json_file_path):
        warnings.warn(colored(f"{json_file_path} does not exist.", "red"))
        return

    try:
        # load json file
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
        # assuming data is a list of dictionaries
        for record in data:
            # the type of record is dict, transfer the dict into str
            context = json.dumps(record)
            # encode the str and write into bin
            write_bin(context, bin_file_path)

        print(colored(f"Successfully converted {json_file_path} to {bin_file_path}", "green"))

    except Exception as e:
        print(colored(f"Error while converting {json_file_path} to {bin_file_path}: {str(e)}", "red"))


def jsonl2bin(jsonl_file_path: str, bin_file_path: str):
    """
    Read content from jsonl file and write to bin file

    Args:
        jsonl_file_path: jsonl file path.
        bin_file_path: bin file path.
    """

    if not os.path.isfile(jsonl_file_path):
        warnings.warn(colored(f"{jsonl_file_path} does not exist.", "red"))
        return

    try:
        with open(jsonl_file_path, "r") as jsonl_file:
            for line in jsonl_file:
                # encode the str and write into bin
                write_bin(line, bin_file_path)

        print(colored(f"Successfully converted {jsonl_file_path} to {bin_file_path}", "green"))

    except Exception as e:
        print(colored(f"Error while converting {jsonl_file_path} to {bin_file_path}: {str(e)}", "red"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_name", required=True, help="Input file name")
    parser.add_argument(
        "--input_file_type",
        choices=["txt", "json", "jsonl"],
        required=True,
        help="Input file format (either txt, json or jsonl)",
    )
    parser.add_argument("--bin", required=True, help="Path to the output bin file")

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    # obtain the raw data path
    input_file_path = f"{args.raw_data_name}.{args.input_file_type}"

    # different methods for different raw data type, we only support "txt", "json" and "jsonl" data type.
    if args.input_file_type == "txt":
        txt2bin(input_file_path, args.bin)
    elif args.input_file_type == "json":
        json2bin(input_file_path, args.bin)
    elif args.input_file_type == "jsonl":
        jsonl2bin(input_file_path, args.bin)
    else:
        print(colored("Invalid input file type. Use --help for more information.", "red"))

    # To avoid potential read/write errors, the metadata preparation follows after creating the .bin file.
    prepare_meta(args.bin)


if __name__ == "__main__":
    main()
