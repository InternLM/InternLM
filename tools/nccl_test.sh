#!/bin/bash

options=$(getopt -o a:b:c:d:e:f:g:h:i:j:k:l: --long launcher:,partition:,nodelist:,gpus_per_node:,nnodes:,ib_threshold:,master_port:,dlc_path:,dlc_config_path:,env_path:,image:,workspace_id: -- "$@")

eval set -- "$options"

while true; do
  case $1 in
    -a | --launcher) shift; launcher=$1 ; shift ;;
    -b | --partition) shift; partition=$1 ; shift ;;
    -c | --nodelist) shift; nodelist=$1 ; shift ;;
    -d | --gpus_per_node) shift; gpus_per_node=$1 ; shift ;;
    -e | --nnodes) shift; nnodes=$1 ; shift ;;
    -f | --ib_threshold) shift; ib_threshold=$1 ; shift ;;
    -g | --master_port) shift; master_port=$1 ; shift ;;
    -h | --dlc_path) shift; dlc_path=$1 ; shift ;;
    -i | --dlc_config_path) shift; dlc_config_path=$1 ; shift ;;
    -j | --env_path) shift; env_path=$1 ; shift ;;
    -k | --image) shift; image=$1 ; shift ;;
    -l | --workspace_id) shift; workspace_id=$1 ; shift ;;
    --) shift; break ;;
    *) echo "Invalid option: $1"; exit 1 ;;
  esac
done

optional_launchers=("slurm" "dlc")
if [[ -z "$launcher" ]]; then
  launcher="slurm"
  echo "Warning: launcher is set to 'slurm'"
fi
if [[ ! "${optional_launchers[@]}" =~ "$launcher" ]]; then
  echo "Error: launcher should in ('slurm' 'dlc')"
  exit 1
fi
if [[ -z "$partition"  && "$launcher" == "slurm" ]]; then
  echo "Error: partition is required if launcher is slurm."
  exit 1
fi
if [[ -z "$gpus_per_node" ]]; then
  gpus_per_node=8
fi
if [[ -z "$nnodes" && -z "$nodelist" ]]; then
  echo "Error: one of nnodes or nodelist should be set"
  exit 1
fi
if [[ -z "$nnodes" && "$launcher" == "dlc" ]]; then
  echo "Error: nnodes should be set if launcher is dlc"
  exit 1
fi
if [[ -z "$ib_threshold" ]]; then
  ib_threshold=25
  echo "Warning: ib_threshold is set to 25"
fi
if [[ -z "$master_port" ]]; then
  master_port=12345
fi
if [[ -z "$dlc_path" && "$launcher" == "dlc" ]]; then
  echo "Error: dlc_path should be set if launcher is dlc"
  exit 1
fi
if [[ -z "$dlc_config_path" && "$launcher" == "dlc" ]]; then
  echo "Error: dlc_config_path should be set if launcher is dlc"
  exit 1
fi
if [[ -z "$env_path" && "$launcher" == "dlc" ]]; then
  echo "Error: env_path should be set if launcher is dlc"
  exit 1
fi
if [[ -z "$image" && "$launcher" == "dlc" ]]; then
  echo "image: nnodes should be set if launcher is dlc"
  exit 1
fi
if [[ -z "$workspace_id" && "$launcher" == "dlc" ]]; then
  echo "workspace_id: nnodes should be set if launcher is dlc"
  exit 1
fi


script_path=$(dirname "$(realpath "$0")")

slow_nodes=""
good_nodes=""
exclude_nodes=""

function do_slurm(){
  local nodes=$1
  local num_nodes=$2
  srun -p $partition -w $nodes -N $num_nodes --gpus-per-task 1 --ntasks-per-node=$gpus_per_node python $script_path/find_slow_nodes.py --launcher slurm --port $master_port --ib_threshold $ib_threshold
}

function do_torch(){
  local master_addr=$1
  torchrun --master_addr $master_addr --master_port $master_port --nnodes $nnodes --nproc_per_node $gpus_per_node $script_path/find_slow_nodes.py --launcher torch --port $master_port --ib_threshold $ib_threshold
}

function do_dlc(){
  local nodes=$1
  local num_nodes=$2
  local master_addr=$3
  local cmd="source ${env_path} && \
cd ${script_path} && \
torchrun --master_addr \$MASTER_ADDR \
--master_port $master_port --nnodes \$WORLD_SIZE \
--nproc_per_node $gpus_per_node --node_rank \$RANK \
find_slow_nodes.py --launcher torch --port $master_port --ib_threshold $ib_threshold"
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
cd ${script_path} && \
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
  if [[ "$launcher" == "slurm" ]] ; then
    do_slurm "$nodestr" "${num_group}"
  elif [[ "$launcher" == "dlc" ]] ; then
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
  if [[ "$slow_nodes" == "" ]] ; then
    exclude_nodes=$slow_nodes
  elif [[ "$good_nodes" == "" ]] ; then
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
      if [[ "$launcher" == "slurm" ]] ; then
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
    local old_exclude_nodes_origin=$(cat exclude_nodes.log)
    rm -f exclude_nodes.log
  fi

  local fininal_exclude_nodes_str="${old_exclude_nodes_origin}${exclude_nodes}"
  echo "exclude_nodes: $fininal_exclude_nodes_str"
  if [[ "$fininal_exclude_nodes_str" != "" ]] ; then
    echo "$fininal_exclude_nodes_str" > exclude_nodes.log
  fi
}

if [[ -z $nodelist ]] ; then
  get_nodes_slurm
else
  nodes=$nodelist
fi

nccl_test "${nodes}"
screen_slow_nodes
echo $exclude_nodes
