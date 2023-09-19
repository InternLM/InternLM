#!/bin/bash
set -x

source ./ci_scripts/common/variables.sh
[[ -n ${GITHUB_WORKSPACE} ]] || { echo "should set GITHUB_WORKSPACE first before ci, exit."; exit 1; }
[[ -n ${CLEAN_PATH} ]] || { echo "should set CLEAN_PATH first before ci, exit."; exit 1; }

readonly CKPTS_PATH="$GITHUB_WORKSPACE/llm_ckpts"
readonly CKPTS20_PATH="$GITHUB_WORKSPACE/llm_ckpts/20"
readonly CKPTS_OUTPUT="${CKPTS20_PATH}/*.pt"
expected_num=22
exit_code=0

source ./ci_scripts/common/basic_func.sh

echo "start to test torch training."

if [[ -d ${CKPTS20_PATH} ]]; then
    if ! rsync -av --remove-source-files ${CKPTS20_PATH} ${CLEAN_PATH}; then
       echo "cleaning cached file in ${CKPTS20_PATH} failed, exit."
       exit 1
    fi
fi

srun -p ${SLURM_PARTITION} --exclusive --quotatype=spot --job-name=$1 -N 1 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29501 train.py --config ./ci_scripts/train/ci_7B_sft.py --launcher torch
[[ $? -ne 0 ]] && { echo "test torch training failed.";  exit_code=$(($exit_code + 1)); }

num=$(num_files "${CKPTS_OUTPUT}")
if [[ ${num} -ne ${expected_num} ]]; then
    echo "expect: ${expected_num} files, actual: ${num} files."
    exit_code=$(($exit_code + 1))
fi

# move the test files.
if ! rsync -av --remove-source-files ${CKPTS_PATH}/* ${CLEAN_PATH}; then
    echo "cleaning cached file in ${CKPTS_PATH} failed."
    exit_code=$(($exit_code + 1))
fi

exit $exit_code
