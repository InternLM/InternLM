#!/bin/bash

launcher=slurm
partition=llm
nodelist=$1

gpus_per_node=8
nnodes=4
master_port=12345


# dlc_path="/cpfs01/shared/public/dlc"
# dlc_config_path="/cpfs01/user/jiaopenglong/.dlc.config"
# env_path="/cpfs01/shared/public/llm/llm-env/llm-env-20230605"
# image="pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/chenxun-st:llm-test"
# workspace_id="ws1so95hgb5kn6ja"

pwd_path=$(pwd)

slow_nodes=""
good_nodes=""
exclude_nodes=""

function do_slurm(){
  local nodes=$1
  local num_nodes=$2
  srun -p $partition -w $nodes -N $num_nodes --gpus-per-task 1 --ntasks-per-node=$gpus_per_node python find_slow_nodes.py --launcher slurm --port $master_port --ib_threshold 25
}

function do_torch(){
  local master_addr=$1
  torchrun --master_addr $master_addr --master_port $master_port --nnodes $nnodes --nproc_per_node $gpus_per_node find_slow_nodes.py --launcher torch --port $master_port
}

function do_dlc(){
  local nodes=$1
  local num_nodes=$2
  local master_addr=$3
  local cmd="source ${env_path} && \
cd ${pwd_path} && \
torchrun --master_addr \$MASTER_ADDR \
--master_port $master_port --nnodes \$WORLD_SIZE \
--nproc_per_node $gpus_per_node --node_rank \$RANK \
find_slow_nodes.py --launcher torch --port $master_port"
  ${dlc_path} --config ${dlc_config_path} create job --name "nccltest" --interactive \
--kind PyTorchJob --node_names $nodes \
--worker_count $num_nodes --worker_cpu 24 --worker_gpu $gpus_per_node \
--worker_memory 100 --worker_image ${image} --workspace_id ${workspace_id} --worker_shared_memory 10Gi \
--command "$cmd"
}

function get_nodes_slurm(){
  if test -f tmp_nccltest.log ; then
    rm -f tmp_nccltest.log
  fi

  local nodestr=$(srun -p $partition -N $nnodes --gpus-per-task $gpus_per_node --ntasks-per-node=1 bash -c 'python find_slow_nodes.py --gethost')
  local nodearray=($(echo $nodestr | awk -F ' ' '{for(i=1; i<=NF; i++) print $i}'))

  nodes=$(printf "%s," "${nodearray[@]}")
  nodes=${nodes%,}
}

function get_nodes_dlc(){
  if test -f test_nodes.log ; then
    rm -f test_nodes.log
  fi

  local cmd="source ${env_path} && \
export DLC_CONFIG=$dlc_config_path && \
cd ${pwd_path} && \
python find_slow_nodes.py --launcher k8s --gethost"

  ${dlc_path} --config ${dlc_config_path} create job --name gethostname --interactive \
--kind PyTorchJob --worker_count $nnodes --worker_cpu 24 --worker_gpu $gpus_per_node \
--worker_memory 100 --worker_image ${image} --workspace_id ${workspace_id} --worker_shared_memory 10Gi \
--command "${cmd}"

  local nodestr=$(cat test_nodes.log)
  local nodearray=($(echo $nodestr | awk -F ' ' '{for(i=1; i<=NF; i++) print $i}'))
  nodes=$(printf "%s," "${nodearray[@]}")
  nodes=${nodes%,}
  rm -f test_nodes.log
}

function nccl_test(){
  local nodestr=$1
  local nodearray=($(echo $nodestr | awk -F ',' '{for(i=1; i<=NF; i++) print $i}'))
  echo $nodestr
  local num_group=${#nodearray[@]}
  if [[ "$launcher" -eq "slurm" ]] ; then
    do_slurm "$nodestr" "${num_group}"
  elif [[ "$launcher" -eq "dlc" ]] ; then
    do_dlc "$nodestr" "${num_group}" "${nodearray[1]}"
  fi

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
      nccl_test "${left_nodestr}"
    fi

    if [[ $num_right_group -le 1 ]] ; then
      slow_nodes="${right_group[@]},$slow_nodes"
    else
      local right_nodestr=$(printf "%s," "${right_group[@]}")
      local right_nodestr=${right_nodestr%,}
      nccl_test "${right_nodestr}"
    fi
  else
    good_nodes="${nodestr},$good_nodes"
  fi
}

function screen_slow_nodes(){
  if [[ "$slow_nodes" -eq "" ]] ; then
    exclude_nodes=$slow_nodes
  elif [[ "$good_nodes" -eq "" ]] ; then
    exclude_nodes=$slow_nodes
  else
    local good_nodearray=($(echo $good_nodes | awk -F ',' '{for(i=1; i<=NF; i++) print $i}'))
    local slow_nodearray=($(echo $slow_nodes | awk -F ',' '{for(i=1; i<=NF; i++) print $i}'))
    local test_good_node=${good_nodearray[1]}
    for slow_node in "${slow_nodearray[@]}"
    do
      local test_nodearray=($slow_node $test_good_node)
      local num_test_nodearray=${#test_nodearray[@]}
      local test_nodestr=$(printf "%s," "${test_nodearray[@]}")
      if [[ "$launcher" -eq "slurm" ]] ; then
        do_slurm "$test_nodestr" "${num_test_nodearray}"
      elif [[ "$launcher" == "dlc" ]] ; then
        do_dlc "$test_nodestr" "${num_test_nodearray}" "${test_nodearray[1]}"
      fi

      if test -f tmp_nccltest.log ; then
        rm -f tmp_nccltest.log
        exclude_nodes="$slow_node,$exclude_nodes"
      fi
    done
  fi

  if test -f exclude_nodes.log ; then
    rm -f exclude_nodes.log
  fi

  if [[ "$exclude_nodes" -ne "" ]] ; then
    echo "$exclude_nodes" > exclude_nodes.log
  fi
}

if [[ -z $nodelist ]] ; then
  get_nodes_slurm
else
  nodes=$nodelist
fi
# nodes="SH-IDC1-10-140-0-153,SH-IDC1-10-140-0-150"
nccl_test "${nodes}"
screen_slow_nodes
echo $exclude_nodes
