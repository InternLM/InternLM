import argparse
import time
import torch
from torch import distributed as dist
import os
import socket
import random

from typing import Dict, List

class NodeInfo:
    def __init__(self, hostname) -> None:
        self.hostname = hostname
        self.ranks = []
        self.nvlink_bw = 0
        self.ib_bw = 0


def get_master_node(launcher:str):
    if launcher == "slurm":
        import subprocess
        result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
        result = result.decode("utf8").strip()
        return result
    elif launcher == "torch":
        return os.environ["MASTER_ADDR"]
    
    
class ProcessFilter:
    def __init__(self, launcher:str, buffer_size: int, ib_threshold: float = 10, ) -> None:
        self.launcher = launcher
        self.ib_threshold = ib_threshold
        self.slow_nodes = []
        self.init_distributed_env()
        self.init_nodeinfo()
        self.buffer_size = buffer_size
        self.buffer_type = torch.float32
        self.type_size = 4
        self.buffer_count = int(self.buffer_size / self.type_size)
        self.buffer = torch.randn((self.buffer_count), dtype=self.buffer_type).cuda(device=f"cuda:{self.local_rank}")
    
    
    def init_distributed_env(self):
        assert self.launcher in ["slurm", "torch"], "launcher only support slurm or torch"
        
        self.master = get_master_node(self.launcher)
        if "MASTER_PORT" in os.environ:
            self.port = os.environ["MASTER_PORT"]
        else:
            self.port = random.randint(20000, 30000)
        self.gpus_per_node = torch.cuda.device_count()
        
        if self.launcher == "slurm":
            self.auto_slurm_env()
        elif self.launcher == "torch":
            self.auto_torch_env()
            
        if 'LOCAL_RANK' not in os.environ:
            self.local_rank = int(os.getenv('RANK')) % self.gpus_per_node
        else:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        
        init_method = f"tcp://[{self.master}]:{self.port}"
        dist.init_process_group(rank=self.rank, world_size=self.world_size, backend="nccl", init_method=init_method)
        

    def auto_slurm_env(self):
        try:
            self.rank = int(os.environ['SLURM_PROCID'])
            self.world_size = int(os.environ['SLURM_NPROCS'])
        except KeyError as e:
            raise RuntimeError(f"Could not find {e} in the SLURM environment")

    def auto_torch_env(self):
        try:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        except KeyError as e:
            raise RuntimeError(f"Could not find {e} in the torch environment")
        
    def init_nodeinfo(self):
        self.nodeinfo = NodeInfo(socket.gethostname())
        self.node_rank_groups = []
        for i in range(0,self.world_size, self.gpus_per_node):
            self.node_rank_groups.append([i + j for j in range(self.gpus_per_node)])
        random.shuffle(self.node_rank_groups)
        
        for rank_group in self.node_rank_groups:
            if self.rank in rank_group:
                self.nodeinfo.ranks = rank_group
                break
        
    def nccl_test(self):
        self.all_reduce_single_node()
        print(f"RANK {self.rank} in HOST {self.nodeinfo.hostname} has internal busBw {self.nodeinfo.nvlink_bw} GB/s")
        self.find_slow_node(self.node_rank_groups)

    def find_slow_node(self, node_rank_groups: List):
        if len(node_rank_groups) == 1:
            print(f"RANK {self.rank} in HOST {self.nodeinfo.hostname} has outer busBw {self.nodeinfo.ib_bw} GB/s")
        
        self.do_all_reduce_multi_nodes(node_rank_groups)
        
        if self.nodeinfo.ib_bw >= self.ib_threshold:
            return
        else:
            left_group, right_group = self.split_node_groups(node_rank_groups)
            self.find_slow_node(left_group)
            self.find_slow_node(right_group)
    
    def split_node_groups(self, node_rank_groups: List):
        number_nodes = len(node_rank_groups)
        
        if number_nodes % 2 == 0:
            return node_rank_groups[: int(number_nodes/2)], node_rank_groups[int(number_nodes/2): ]
        else:
            return node_rank_groups[: int(number_nodes/2) + 1], node_rank_groups[int(number_nodes/2): ]
        
    def all_reduce_single_node(self):
        self.nodeinfo.nvlink_bw = self.do_reduce(self.nodeinfo.ranks)
        
    def do_all_reduce_multi_nodes(self, node_rank_groups: List):
        ranks = []
        for node_rank_group in node_rank_groups:
            ranks.extend(node_rank_group)
        self.nodeinfo.ib_bw = self.do_all_reduce(ranks)
            
    def do_all_reduce(self, ranks: List):
        group = dist.new_group(ranks)
        
        dist.barrier(group=group)
        s = time.time()
        dist.all_reduce(self.buffer, group=group, op = dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        cost = time.time() - s
        dist.barrier(group=group)
        
        bus_bw = self.get_bus_bandwidth(cost)
        return bus_bw
        

    def get_bus_bandwidth(self, cost_seconds:float):
        base_bw = self.buffer_count * self.world_size * self.type_size / 1e9 / cost_seconds
        factor = (self.world_size - 1) / self.world_size
        
        bus_bw = base_bw * factor
        
        return bus_bw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nccl test tools for finding slow nodes")
    parser.add_argument("--launcher", type=str, help="launcher (slurm or torch)")
    parser.add_argument("--threshold", type=float, default=10, help="launcher (slurm or torch)")
    parser.add_argument("--buffersize", type=int, default=512*1014*1024, help="launcher (slurm or torch)")
    
    args = parser.parse_args()
    filter = ProcessFilter(launcher=args.launcher, ib_threshold=args.threshold, buffer_size=args.buffersize)
    
    filter.nccl_test()
    