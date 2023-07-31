#!/bin/bash

export exit_code=0

function if_exist() {
    ls -l $file_path
    exit_code_now=$?
    exit_code=$(($exit_code + $exit_code_now))
}

function num_files() {
    file_num=$(ls -l $file_dir |wc -l)
    echo "there are $file_num files in $file_dir"
}
