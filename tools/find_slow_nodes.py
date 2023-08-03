# coding:utf-8
import argparse
import copy
import fcntl
import time
import torch
from torch import distributed as dist
import os
import socket
import re
import base64
import hashlib
import hmac
import uuid
import requests

from datetime import datetime
from typing import List
from configparser import ConfigParser

def printflock(*msgs,flush: bool=False):
    """ print """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs, flush=flush)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

class DLCJobInfo:
    """
    get dlc job detailed information
    """

    def __init__(
        self,
        endpoint,
        access_key_id,
        access_key_secret,
        protocol,
    ):
        self.endpoint = endpoint
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.protocol = protocol

    def _get_canonicalized_resource(self, pathname, query):
        if len(query) <= 0:
            return pathname
        resource = f"{pathname}?"
        query_list = sorted(list(query))
        for key in query_list:
            if query[key] is not None:
                if query[key] == "":
                    s = f"{key}&"
                else:
                    value = self.to_string(query[key])
                    s = f"{key}={value}&"
            resource += s
        return resource[:-1]

    def to_string(self, s, encoding="utf-8"):
        if s is None:
            return s
        if isinstance(s, bytes):
            return s.decode(encoding)
        else:
            return str(s)

    def get_string_to_sign(self, method, url_path, headers, query):
        """使用请求信息生成待签名的字符串"""
        accept = "" if headers.get("accept") is None else headers.get("accept")
        content_md5 = "" if headers.get("content-md5") is None else headers.get("content-md5")
        content_type = "" if headers.get("content-type") is None else headers.get("content-type")
        date = "" if headers.get("date") is None else headers.get("date")
        header = f"{method}\n{accept}\n{content_md5}\n{content_type}\n{date}\n"

        canon_headers = self._get_canonicalized_headers(headers)
        canon_resource = self._get_canonicalized_resource(url_path, query)
        sign_str = header + canon_headers + canon_resource
        return sign_str

    def get_roasignature(self, string_to_sign, secret):
        """生成签名: 使用HMAC-256生成签名, 然后通过base64输出签名字符串。"""
        hash_val = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
        signature = base64.b64encode(hash_val).decode("utf-8")
        return signature

    def _get_canonicalized_headers(self, headers):
        """将请求头排序后, 获取“x-acs-”作为前缀按字母序排序后拼接。
        注意, 按RFC2616, HTTP 请求头的名称是大小写不敏感的。
        """
        canon_keys = []
        for k in headers:
            if k.startswith("x-acs-"):
                canon_keys.append(k)
        canon_keys = sorted(canon_keys)
        canon_header = ""
        for k in canon_keys:
            canon_header += f"{k}:{ headers[k]}\n"
        return canon_header

    def do_request(self, api_product, api_query, api_method, api_path):
        """根据请求信息，生成认证信息，发送请求给到后端服务"""
        ts = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        signature_nonce = str(uuid.uuid4())
        headers = {
            "x-acs-signature-method": "HMAC-SHA1",
            "date": ts,
            "x-acs-signature-nonce": signature_nonce,
            "x-pai-product": api_product,
            "accept": "application/json",
        }
        api_url = f"{self.protocol}://{self.endpoint}{api_path}"
        # 使用请求信息，生成请求使用的签名(signature)，然后生成对应认证信息，在请求头里传递给到服务(authorization)
        string_to_sign = self.get_string_to_sign(method=api_method, url_path=api_path, headers=headers, query=api_query)
        signature = self.get_roasignature(string_to_sign=string_to_sign, secret=self.access_key_secret)
        headers["authorization"] = f"acs {self.access_key_id}:{signature}"
        resp = requests.request(method=api_method, url=api_url, params=api_query, headers=headers, verify=False)

        # print(resp.status_code)
        # print(resp.content)

        if resp.status_code != 200:
            print(resp.text)
            return None
        return resp.json()

    def do_get_job(self, jobid: str):
        # API请求的URL，以及参数
        api_path = f"/api/v1/jobs/{jobid}"
        # 填写请求query上的参数
        api_method = "GET"
        api_product = "dlc"
        api_query = {"jobid": jobid}

        job_meta = self.do_request(
            api_product=api_product, api_query=api_query, api_method=api_method, api_path=api_path
        )

        return job_meta

    def get_node_pod(self, podmetalist):
            nodelist = []
            podlist = []

            for podmeta in podmetalist:
                nodelist.append(podmeta["NodeName"])
                podlist.append(podmeta["PodId"])

            return list(set(nodelist)), list(set(podlist))

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
        
        #print("start init process group", flush=True)
        init_method = f"tcp://[{self.master}]:{self.port}"
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=self.world_size, rank=self.rank)

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
        if self.rank == 0:
            printflock(f"RANK {self.rank} in HOST {self.nodeinfo.hostname} has outer busBw {self.nodeinfo.ib_bw} GB/s", flush=True)
        dist.destroy_process_group()
        
        if self.nodeinfo.ib_bw < self.ib_threshold and self.local_rank == 0:
            with open("tmp_nccltest.log", "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(f"{self.nodeinfo.hostname} has busBw  {self.nodeinfo.ib_bw} GB/s \n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            
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
        printflock(f"RANK {self.rank} in HOST {self.nodeinfo.hostname} has internal busBw {self.nodeinfo.nvlink_bw} GB/s", flush=True)
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
    parser.add_argument("--launcher", type=str, default="slurm", help="launcher (slurm, k8s, torch)")
    parser.add_argument("--ib_threshold", type=float, default=18, help="IB bandwidth threshold")
    parser.add_argument("--nvlink_threshold", type=float, default=200, help="NVLink bandwidth threshold")
    parser.add_argument("--buffersize", type=int, default=2*1024*1014*1024, help="test tensor buffer size")
    parser.add_argument("--port", type=int, default=29500, help="torch distributed port")
    parser.add_argument("--local", action="store_true", help="nccl test on per node")
    parser.add_argument("--warmup", type=int, default=5, help="warm-up iters")
    parser.add_argument("--iters", type=int, default=20, help="communication iters")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gethost", action="store_true")
    
    args = parser.parse_args()
    if args.gethost:
        if args.launcher == "k8s":
            jobid = os.getenv("KUBERNETES_POD_NAME").split("-")[0]
            conf = ConfigParser()
            conf.read(os.getenv("DLC_CONFIG"))
            dlcjob_info = DLCJobInfo(
                conf["aliyun"]["endpoint"].replace('"', ''),
                conf["user"]["access_id"].replace('"', ''),
                conf["user"]["access_key"].replace('"', ''),
                conf["aliyun"]["protocol"].replace('"', ''),
            )
            jobmeta=dlcjob_info.do_get_job(jobid)
            nodelist, podlist = dlcjob_info.get_node_pod(jobmeta["Pods"])
            nodeliststr = " ".join(nodelist)
            with open("test_nodes.log", "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(f"{nodeliststr}")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            print(socket.gethostname())
    else:
        filter = ProcessFilter(launcher=args.launcher, ib_threshold=args.ib_threshold, nvlink_threshold=args.nvlink_threshold, buffer_size=args.buffersize, port=args.port, is_local=args.local, warmup=args.warmup, iters=args.iters)
        
        filter.nccl_test()
    
