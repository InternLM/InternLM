#!/bin/bash

rm -rf /mnt/petrelfs/qa-caif-cicd/data/lm_data/cn_data/result.*
srun -p llm python tools/tokenizer.py --text_input_path /mnt/petrelfs/qa-caif-cicd/data/lm_data/cn_data/raw_data.txt --bin_output_path /mnt/petrelfs/qa-caif-cicd/data/lm_data/cn_data/result.bin

file_one="$DIR_PREFIX/data/lm_data/cn_data/result.bin"
file_two="$DIR_PREFIX/data/lm_data/cn_data/result.bin.meta"
file_list=($file_one $file_two)

source ./ci_scripts/common/basic_func.sh
for file_path in ${file_list[@]};
do
    if_exist $file_path
done

if [ $exit_code -ne 0 ]
then
    exit 1
fi
