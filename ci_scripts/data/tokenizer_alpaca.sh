#!/bin/bash

rm -rf /mnt/petrelfs/qa-caif-cicd/data/lm_data/alpaca_data/result/*

python tools/alpaca_tokenizer.py /mnt/petrelfs/qa-caif-cicd/data/lm_data/alpaca_data/alpaca_data.json /mnt/petrelfs/qa-caif-cicd/data/lm_data/alpaca_data/result  tools/V7_sft.model --split_ratio 0.1

file_one="/mnt/petrelfs/qa-caif-cicd/data/lm_data/alpaca_data/result/train/en/dataset.bin"
file_two="/mnt/petrelfs/qa-caif-cicd/data/lm_data/alpaca_data/result/train/en/dataset.bin.meta"
file_three="/mnt/petrelfs/qa-caif-cicd/data/lm_data/alpaca_data/result/valid/en/dataset.bin"
file_four="/mnt/petrelfs/qa-caif-cicd/data/lm_data/alpaca_data/result/valid/en/dataset.bin.meta"
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
