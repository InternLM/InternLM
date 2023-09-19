#!/bin/bash
set -x

source ./ci_scripts/common/variables.sh
[[ -n ${DATA_VOLUME} ]] || { echo "should set DATA_VOLUME first before ci, exit."; exit 1; }
[[ -n ${CLEAN_PATH} ]] || { echo "should set CLEAN_PATH first before ci, exit."; exit 1; }

readonly DATA=${DATA_VOLUME}/lm_data/cn_data/raw_data.txt
readonly RESULT=${DATA_VOLUME}/lm_data/cn_data/result.bin
readonly RESULT_META=${DATA_VOLUME}/lm_data/cn_data/result.bin.meta
readonly RESULTS=${DATA_VOLUME}/lm_data/cn_data/result.*
exit_code=0

source ./ci_scripts/common/basic_func.sh

echo "start to test tokenizer.py."

num=$(num_files "${RESULTS}")
if [[ ${num} -gt 0 ]]; then
    if ! rsync -av --remove-source-files ${RESULTS} ${CLEAN_PATH}; then
       echo "cleaning test data ${RESULTS} failed, exit."
       exit 1
    fi
fi

srun -p ${SLURM_PARTITION} --quotatype=spot --job-name=$1 --gpus-per-task=1 python tools/tokenizer.py --text_input_path ${DATA} --bin_output_path ${RESULT}
[[ $? -ne 0 ]] && { echo "test tokenizer.py failed.";  exit_code=$(($exit_code + 1)); }

file_list=($RESULT $RESULT_META)
for file in ${file_list[@]}; do
    if [[ ! -f ${file} ]]; then
        echo "expect: ${file} exists, actual: not exist."
        exit_code=$(($exit_code + 1))
    fi
done

# move the test files.
if ! rsync -av --remove-source-files ${RESULTS} ${CLEAN_PATH}; then
   echo "cleaning cached file in ${RESULTS} failed."
   exit_code=$(($exit_code + 1))
fi

exit $exit_code
