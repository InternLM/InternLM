#!/bin/bash
set -x

source ./ci_scripts/common/variables.sh
[[ -n ${DATA_VOLUME} ]] || { echo "should set DATA_VOLUME first before ci, exit."; exit 1; }
[[ -n ${CLEAN_PATH} ]] || { echo "should set CLEAN_PATH first before ci, exit."; exit 1; }

readonly SRC_DATASET_META=${DATA_VOLUME}/lm_data/alpaca_data/alpaca_data.json
readonly RESULTS=${DATA_VOLUME}/lm_data/alpaca_data/result
readonly TRAIN_DATASET=${RESULTS}/train/en/dataset.bin
readonly TRAIN_DATASET_META=${RESULTS}/train/en/dataset.bin.meta
readonly VALID_DATASET=${RESULTS}/valid/en/dataset.bin
readonly VALID_DATASET_META=${RESULTS}/valid/en/dataset.bin.meta

split_ratio=0.1
exit_code=0

source ./ci_scripts/common/basic_func.sh

echo "start to test alpaca_tokenizer.py."

if [[ -d ${RESULTS} ]]; then
    if ! rsync -av --remove-source-files ${RESULTS} ${CLEAN_PATH}; then
       echo "cleaning test data in ${RESULTS} failed, exit."
       exit 1
    fi
fi

if [[ ! -f ${SRC_DATASET_META} ]]; then
   echo "${SRC_DATASET_META} should be exist, exit."
   exit 1
fi

python tools/alpaca_tokenizer.py ${SRC_DATASET_META} ${RESULTS} tools/V7_sft.model --split_ratio ${split_ratio}
[[ $? -ne 0 ]] && { echo "test alpaca_tokenizer.py failed.";  exit_code=$(($exit_code + 1)); }

file_list=(${TRAIN_DATASET} ${TRAIN_DATASET_META} ${VALID_DATASET} ${VALID_DATASET_META})
for file in ${file_list[@]}; do
    if [[ ! -f ${file} ]]; then
        echo "expect: ${file} exists, actual: not exist."
        exit_code=$(($exit_code + 1))
    fi
done

# move the test files.
if ! rsync -av --remove-source-files ${RESULTS} ${CLEAN_PATH}; then
    echo "cleaning test data in ${RESULTS} failed."
    exit_code=$(($exit_code + 1))
fi

exit $exit_code
