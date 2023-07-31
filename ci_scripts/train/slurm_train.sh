#!/bin/bash

rm -rf $GITHUB_WORKSPACE/llm_ckpts/20

srun -p llm2 --quotatype=spot -n 8 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./ci_scripts/train/ci_7B_sft.py

file_dir="$GITHUB_WORKSPACE/llm_ckpts/20/*.pt"
source ./ci_scripts/common/basic_func.sh

num_files ${file_dir}

if [ $file_num -ne 22 ]
then
    echo "The num of files is not right"
    ls -l $file_dir
    rm -rf $GITHUB_WORKSPACE/llm_ckpts
    exit 1
fi


