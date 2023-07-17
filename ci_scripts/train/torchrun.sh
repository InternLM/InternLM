#!/bin/bash

echo "train demo by torchrun"
#torchrun --nnodes=1 --nproc_per_node=8 train.py --config ./configs/ci_7B_sft.py --launcher "torch"

