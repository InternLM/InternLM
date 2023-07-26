#!/bin/bash

launcher=slurm
partition=llm2

gpus_per_node=1
nnodes=4
master_port=12345

slow_nodes=""

function do_slurm(){
  local nodes=$1
  local num_nodes=$2
  srun -p $partition -w $nodes -N $num_nodes --gpus-per-task 1 --ntasks-per-node=$gpus_per_node python find_slow_nodes.py --launcher slurm --port $master_port
}

function get_nodes_slurm(){
    nodes=$(srun -p $partition -N $nnodes --gpus-per-task 1 --ntasks-per-node=$gpus_per_node bash -c 'python find_slow_nodes.py --launcher slurm --nodes $SLURM_NODELIST')
}

function do_torch(){
    local master_addr=$1
    torchrun --master_addr $master_addr --master_port $master_port --nnodes $nnodes --nproc_per_node $gpus_per_node find_slow_nodes.py --launcher torch --port $master_port
}

function nccl_test_slurm(){
  local nodestr=$1
  IFS=',' read -ra nodearray <<< "$nodestr"
  echo $nodestr
  local num_group=${#nodearray[@]}
  do_slurm "$nodestr" "${num_group}"
  if test -f tmp_nccltest.log ; then
    rm -f tmp_nccltest.log
    local middle=$((num_group / 2))
    if ((length % 2 != 0)); then
        local middle=$((middle + 1))
    fi
    local left_group=("${nodearray[@]:0:middle}")
    local right_group=("${nodearray[@]:middle}")

    #split_node "${nodearray[@]}"
    echo "left_group ${left_group[@]}"
    echo "right_group ${right_group[@]}"
    local num_left_group=${#left_group[@]}
    local num_right_group=${#right_group[@]}
    if [[ $num_left_group -le 1 ]] ; then
      slow_nodes="${left_group[@]},$slow_nodes"
    else
      local left_nodestr=$(printf "%s," "${left_group[@]}")
      local left_nodestr=${left_nodestr%,}
      nccl_test_slurm "${left_nodestr}"
    fi

    if [[ $num_right_group -le 1 ]] ; then
      slow_nodes="${right_group[@]},$slow_nodes"
    else
      local right_nodestr=$(printf "%s," "${right_group[@]}")
      local right_nodestr=${right_nodestr%,}
      nccl_test_slurm "${right_nodestr}"
    fi
  fi
}

get_nodes_slurm
nccl_test_slurm "${nodes}"

# echo $slow_nodes
