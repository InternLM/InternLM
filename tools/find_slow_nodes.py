import argparse
import time
import torch
from torch import distributed as dist
import os
import socket

from typing import List

class NodeInfo:
    def __init__(self, hostname) -> None:
        self.hostname = hostname
        self.host_encoding = torch.Tensor([ord(i) for i in self.hostname]).type(torch.int)
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
    def __init__(self, launcher:str, buffer_size: int, ib_threshold: float, nvlink_threshold: float, port: int, is_local: bool, warmup: int, iters: int) -> None:
        self.launcher = launcher
        self.port = port
        self.ib_threshold = ib_threshold
        self.nvlink_threshold = nvlink_threshold
        self.is_local = is_local
        self.warmup = warmup
        self.iters = iters
        
        self.slow_nodes = []
        self.nodeinfo = NodeInfo(socket.gethostname())
        
        self.buffer_size = buffer_size
        self.buffer_type = torch.float32
        self.type_size = 4
        self.buffer_count = int(self.buffer_size / self.type_size)
        self.buffer = torch.randn((self.buffer_count), dtype=self.buffer_type)
        
    
    def init_distributed_env(self, master = None ):
        assert self.launcher in ["slurm", "torch"], "launcher only support slurm or torch"
        
        if not master:
            self.master = get_master_node(self.launcher)
        else:
            self.master = master
        self.gpus_per_node = torch.cuda.device_count()
        
        if self.launcher == "slurm":
            self.auto_slurm_env()
        elif self.launcher == "torch":
            self.auto_torch_env()
            
        if 'LOCAL_RANK' not in os.environ:
            self.local_rank = self.rank % self.gpus_per_node
        else:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        
        #print(self.rank, self.local_rank, self.world_size, self.master, self.port, flush=True )
        # os.environ['MASTER_PORT'] = str(self.port)
        # os.environ['LOCAL_RANK'] = str(self.local_rank)
        # os.environ['WORLD_SIZE'] = str(self.world_size)
        # os.environ['RANK'] = str(self.rank)
        # os.environ['MASTER_ADDR'] = self.master
        
        
        # init_method = "env://"
        
        # print("start init process group", flush=True)
        init_method = f"tcp://[{self.master}]:{self.port}"
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=self.world_size, rank=self.rank) #

    def auto_slurm_env(self):
        try:
            if self.is_local:
                self.rank = int(os.environ['SLURM_PROCID']) % self.gpus_per_node
                self.world_size = self.gpus_per_node
            else:
                self.rank = int(os.environ['SLURM_PROCID'])
                self.world_size = int(os.environ['SLURM_NPROCS'])
        except KeyError as e:
            raise RuntimeError(f"Could not find {e} in the SLURM environment")

    def auto_torch_env(self):
        try:
            if self.is_local:
                self.rank = int(os.environ["RANK"]) % self.gpus_per_node
                self.world_size = self.gpus_per_node
            else:
                self.rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])
        except KeyError as e:
            raise RuntimeError(f"Could not find {e} in the torch environment")
        
    def init_nodeinfo(self):
        self.node_rank_groups = []
        for i in range(0,self.world_size, self.gpus_per_node):
           self.node_rank_groups.append([i + j for j in range(self.gpus_per_node)])
        
        for rank_group in self.node_rank_groups:
            if self.rank in rank_group:
                self.nodeinfo.ranks = rank_group
                break
        
        
    def nccl_test(self):
        if self.is_local:
            self.do_all_reduce_single_node()
        else:
            self.do_all_reduce_multi_node()

    def do_all_reduce_multi_node(self):
        self.init_distributed_env()
        self.init_nodeinfo()
        self.buffer = self.buffer.cuda(device=f"cuda:{self.local_rank}")
        ranks = []
        for node_rank_group in self.node_rank_groups:
            ranks.extend(node_rank_group)
        group = dist.new_group(ranks)
        count = 1
        sum = 0
        while count <= self.iters:
            bw = self.do_all_reduce(ranks,group=group)
            count += 1
            if count > self.warmup:
                sum += bw
        self.nodeinfo.ib_bw = sum / (self.iters - self.warmup)
        print(f"RANK {self.rank} in HOST {self.nodeinfo.hostname} has outer busBw {self.nodeinfo.ib_bw} GB/s", flush=True)
        dist.destroy_process_group()
        
        with open("tmp_nccltest.log", "a+") as f:
            if self.nodeinfo.ib_bw < self.ib_threshold and self.local_rank == 0:
                f.write(f"{self.nodeinfo.hostname} has busBw  {self.nodeinfo.ib_bw} GB/s \n")
            
    def do_all_reduce_single_node(self):
        self.init_distributed_env(master=self.nodeinfo.hostname)
        self.init_nodeinfo()
        #print("node info", self.nodeinfo.hostname, self.nodeinfo.ranks, flush=True)
        self.buffer = self.buffer.cuda(device=f"cuda:{self.local_rank}")
        group = dist.new_group(self.nodeinfo.ranks)
        count = 1
        sum = 0
        while count <= self.iters:
            bw = self.do_all_reduce(self.nodeinfo.ranks,group=group)
            if count > self.warmup:
                sum += bw
            count += 1
        self.nodeinfo.nvlink_bw = sum / (self.iters - self.warmup)
        print(f"RANK {self.rank} in HOST {self.nodeinfo.hostname} has internal busBw {self.nodeinfo.nvlink_bw} GB/s", flush=True)
        dist.destroy_process_group()
        
            
    def do_all_reduce(self, ranks: List, group):
        dist.barrier(group=group)
        s = time.time()
        dist.all_reduce(self.buffer, group=group, op = dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        cost = time.time() - s
        dist.barrier(group=group)
        
        bus_bw = self.get_bus_bandwidth(cost, len(ranks))
        return bus_bw
        

    def get_bus_bandwidth(self, cost_seconds:float, ntasks: int):
        #print(f"self.buffer_count: {self.buffer_count}, self.type_size: {self.type_size}, cost_seconds: {cost_seconds}, ntasks: {ntasks}")
        base_bw = self.buffer_count * self.type_size / 1e9 / cost_seconds 
        factor = 2 * (ntasks - 1) / ntasks
        
        bus_bw = base_bw * factor
        
        return bus_bw
    
    def split_node_groups(self, node_rank_groups: List):
        number_nodes = len(node_rank_groups)
        
        if number_nodes % 2 == 0:
            left_group, right_group  = node_rank_groups[: int(number_nodes/2)], node_rank_groups[int(number_nodes/2): ]
            return left_group, right_group
        else:
            left_group, right_group = node_rank_groups[: int(number_nodes/2) + 1], node_rank_groups[int(number_nodes/2)+1: ]
            return left_group, right_group


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nccl test tools for finding slow nodes")
    parser.add_argument("--launcher", type=str, help="launcher (slurm or torch)")
    parser.add_argument("--ib_threshold", type=float, default=18, help="IB bandwidth threshold")
    parser.add_argument("--nvlink_threshold", type=float, default=200, help="NVLink bandwidth threshold")
    parser.add_argument("--buffersize", type=int, default=512*1014*1024, help="test tensor buffer size")
    parser.add_argument("--port", type=int, default=29500, help="torch distributed port")
    parser.add_argument("--local", action="store_true", help="nccl test on per node")
    parser.add_argument("--warmup", type=int, default=5, help="warm-up iters")
    parser.add_argument("--iters", type=int, default=20, help="communication iters")
    
    
    args = parser.parse_args()
    filter = ProcessFilter(launcher=args.launcher, ib_threshold=args.ib_threshold, nvlink_threshold=args.nvlink_threshold, buffer_size=args.buffersize, port=args.port, is_local=args.local, warmup=args.warmup, iters=args.iters)
    
    filter.nccl_test()
    
