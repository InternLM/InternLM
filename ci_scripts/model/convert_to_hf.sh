#!/bin/bash
set -x

source ./ci_scripts/common/variables.sh
[[ -n ${DATA_VOLUME} ]] || { echo "should set DATA_VOLUME first before ci, exit."; exit 1; }
[[ -n ${GITHUB_WORKSPACE} ]] || { echo "should set GITHUB_WORKSPACE first before ci, exit."; exit 1; }
[[ -n ${CLEAN_PATH} ]] || { echo "should set CLEAN_PATH first before ci, exit."; exit 1; }

readonly CKPTS_INPUT="${DATA_VOLUME}/lm_data/alpaca_data/llm_ckpts/20"
readonly CKPTS_OUTPUT="${GITHUB_WORKSPACE}/hf_ckpt"
readonly TOKENIZER="${GITHUB_WORKSPACE}/hf_ckpt/tokenizer.model"
readonly CONFIG="${GITHUB_WORKSPACE}/hf_ckpt/config.json"
readonly INERNLM="${GITHUB_WORKSPACE}/hf_ckpt/modeling_internlm.py"
exit_code=0
expected_num=9

source ./ci_scripts/common/basic_func.sh

echo "start to test convert2hf.py."

if [[ -d ${CKPTS_OUTPUT} ]]; then
    if ! rsync -av --remove-source-files ${CKPTS_OUTPUT}/* ${CLEAN_PATH}; then
       echo "cleaning cached file in ${CKPTS_OUTPUT} failed, exit."
       exit 1
    fi
fi

python ./tools/transformers/convert2hf.py --src_folder ${CKPTS_INPUT} --tgt_folder ${CKPTS_OUTPUT} --tokenizer ./tools/V7_sft.model
[[ $? -ne 0 ]] && { echo "test convert2hf.py failed.";  exit_code=$(($exit_code + 1)); }

#assert exists model
file_list=($TOKENIZER $CONFIG $INERNLM)
for file in ${file_list[@]}; do
    if [[ ! -f ${file} ]];then
        echo "file ${file} does not exist."
        exit_code=$(($exit_code + 1))
    fi
done

num=$(num_files "${CKPTS_OUTPUT}")

if [[ ${num} -ne ${expected_num} ]]; then
    echo "expect: ${expected_num} files, actual: ${num} files."
    exit_code=$(($exit_code + 1))
fi

# NOTICE: should not remove the cached files, because the cached files will be used in the next test case.
exit $exit_code
