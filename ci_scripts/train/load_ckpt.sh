#!/bin/bash
set -x

source ./ci_scripts/common/variables.sh
[[ -n ${GITHUB_WORKSPACE} ]] || { echo "should set GITHUB_WORKSPACE first before ci, exit."; exit 1; }
[[ -n ${CLEAN_PATH} ]] || { echo "should set CLEAN_PATH first before ci, exit."; exit 1; }

readonly CKPTS_PATH="$GITHUB_WORKSPACE/llm_ckpts"
readonly CKPTS40_PATH="$GITHUB_WORKSPACE/llm_ckpts/40"
readonly CKPTS40_OUTPUT="${CKPTS40_PATH}/*.pt"
expected_num=22
exit_code=0

source ./ci_scripts/common/basic_func.sh

echo "start to test slurm training with loading checkpoint."

python ./ci_scripts/train/generate_config.py --case_name $1
file="./ci_scripts/train/$1.py"
if [[ ! -f ${file} ]]; then
        echo "expect: ${file} exists, actual: not exist."
        exit_code=$(($exit_code + 1))
    fi

srun -p ${SLURM_PARTITION} --exclusive --quotatype=spot --job-name=$2 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ${file}
[[ $? -ne 0 ]] && { echo "test slurm training failed.";  exit_code=$(($exit_code + 1)); }


num=$(num_files "${CKPTS40_OUTPUT}")
if [[ ${num} -ne ${expected_num} ]]; then
    echo "expect: ${expected_num} files, actual: ${num} files."
    exit_code=$(($exit_code + 1))
fi

# move the test files.
if [[ -d ${CKPTS_PATH} ]]; then
    if ! rsync -av --remove-source-files ${CKPTS_PATH} ${CLEAN_PATH}; then
        echo "cleaning cached file in ${CKPTS_PATH} failed."
        exit_code=$(($exit_code + 1))
    fi
fi

exit $exit_code
