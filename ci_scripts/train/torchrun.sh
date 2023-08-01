#!/bin/bash

rm -rf $GITHUB_WORKSPACE/llm_ckpts/20
srun -p llm2 -N 1 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29501 train.py --config ./ci_scripts/train/ci_7B_sft.py --launcher "torch"

file_dir="$GITHUB_WORKSPACE/llm_ckpts/20/*.pt"
source ./ci_scripts/common/basic_func.sh

num_files ${file_dir}

if [ $file_num -ne 21 ]
then
    echo "The num of files is not right"
    ls -l $file_dir
    rm -rf $GITHUB_WORKSPACE/llm_ckpts
    exit 1
fi
