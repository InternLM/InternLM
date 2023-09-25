#!/bin/bash

readonly DATA_VOLUME=$(echo $GITHUB_WORKSPACE | cut -d '/' -f 1-4)/data
readonly CLEAN_PATH=$(echo $GITHUB_WORKSPACE | cut -d '/' -f 1-4)/ci_clean_bak
