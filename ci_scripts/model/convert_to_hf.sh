#!/bin/bash

rm -rf ./hf_ckpt/*
python ./tools/transformers/convert2hf.py --src_folder $DIR_PREFIX/data/lm_data/alpaca_data/llm_ckpts/20 --tgt_folder hf_ckpt/ --tokenizer ./tools/V7_sft.model

#assert exists model
file_one="$GITHUB_WORKSPACE/hf_ckpt/tokenizer.model"
file_two="$GITHUB_WORKSPACE/hf_ckpt/config.json"
file_three="$GITHUB_WORKSPACE/hf_ckpt/modeling_internlm.py"
file_list=($file_one $file_two $file_three)
file_dir="$GITHUB_WORKSPACE/hf_ckpt/*"

source ./ci_scripts/common/basic_func.sh

for file_path in ${file_list[@]};
do
    if_exist $file_path
done


num_files ${file_dir}

if [ $file_num -ne 9 ]
then
    echo "The num of files is not right"
    ls -l $file_dir
    exit_code=$(($exit_code + 1)) 
fi

if [ $exit_code -ne 0 ]
then
    exit 1
fi
