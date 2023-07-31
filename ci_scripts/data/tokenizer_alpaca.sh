#!/bin/bash

rm -rf $DIR_PREFIX/data/lm_data/alpaca_data/result/*

python tools/alpaca_tokenizer.py $DIR_PREFIX/data/lm_data/alpaca_data/alpaca_data.json $DIR_PREFIX/data/lm_data/alpaca_data/result  tools/V7_sft.model --split_ratio 0.1

file_one="$DIR_PREFIX/data/lm_data/alpaca_data/result/train/en/dataset.bin"
file_two="$DIR_PREFIX/data/lm_data/alpaca_data/result/train/en/dataset.bin.meta"
file_three="$DIR_PREFIX/data/lm_data/alpaca_data/result/valid/en/dataset.bin"
file_four="$DIR_PREFIX/data/lm_data/alpaca_data/result/valid/en/dataset.bin.meta"
file_list=($file_one $file_two $file_three $file_four)

source ./ci_scripts/common/basic_func.sh
for file_path in ${file_list[@]};
do
    if_exist $file_path
done

if [ $exit_code -ne 0 ]
then
    exit 1
fi
