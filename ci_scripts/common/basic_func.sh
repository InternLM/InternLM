#!/bin/bash

#######################################
# Calculate the number of files in a directory.
# Call this function like this: num_files "${file_path}".
# Globals:
#   None
# Arguments:
#   $1: the directory path
# Returns:
#   the number of files in the directory
#######################################
num_files() {
    [[ $# -eq 1 ]] || return 1
    local file_num
    file_num=$(ls -l $1 | grep '^-' | wc -l)
    echo $file_num
}
