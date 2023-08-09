import os
import socket
import time
from collections import defaultdict
from typing import Dict

import requests
import torch

from internlm.utils.logger import get_logger

from .utils import get_job_id, get_world_size

logger = get_logger(__file__)

local_adapter = None


class Adapter(object):
    """Adapter"""

    def __init__(self, rank: int, local_rank: int, cfg: str, coordinator_ip: str = None, coordinator_port: int = None):
        """
        Adapter acts as a client, each rank will initialize one, and will send registration information,
        heartbeat, and exception information to the Coordinator process.

        Args:
            rank (int): the global rank of the current rank
            local_rank (int): the local rank of the current rank
            cfg (str): the config of train
            coordinator_ip (str): the ip of the coordinator
            coordinator_port (int): the port of the coordinator
        """

        self._worker_addr = defaultdict(list)
        self._coordinator_ip = coordinator_ip if coordinator_ip else os.environ["COORDIATOR_IP"]
        self._coordinator_port = coordinator_port if coordinator_port else os.environ["COORDIATOR_PORT"]
        self._coordinator_url_prefix = "http://" + self._coordinator_ip + ":" + str(self._coordinator_port) + "/"

        self._rank = rank
        self._local_rank = local_rank
        self._ip = socket.gethostbyname(socket.gethostname())
        self._last_update_request_addr_time = time.time()
        self._cfg = str(cfg)

        self._register()
        # After more than fifteen minutes, try to reconnect to the Coordinator
        # self.last_active_time = time.time()

    def _send_requests(self, meta_data: Dict, msg_type: str, show_log=False, timeout=10):
        result = None
        max_retry = 3
        while True:
            if max_retry == 0:
                break
            try:
                headers = {"Accept": "application/json", "Content-Type": "application/json"}
                response = requests.post(
                    self._coordinator_url_prefix + f"coordinator/{msg_type}",
                    headers=headers,
                    json=meta_data,
                    timeout=timeout,
                ).json()
                if response["code"] == 0:
                    if show_log:
                        logger.info(f"{msg_type} adapter successfully, rank: {self._rank}, ip: {self._ip}")
                    result = response["info"]
                    # self.last_active_time = time.time()
                    break
                else:
                    max_retry -= 1
                    if show_log:
                        logger.error(f"{msg_type} adapter failed, rank: {self._rank}")
                        break
            except Exception as e:
                max_retry -= 1
                logger.error(f"[Network ERROR] coordinator error: {e}")

        return result

    def _register(self):
        # Coordinator does not need to send messages to adapter.
        # Each rank sends all the information, there is some redundancy, but do it first.
        meta_data = {
            "rank": self._rank,
            "hostname": socket.gethostname(),
            "device_id": self._local_rank,
            "slurm_jobid": get_job_id(),
            "slurm_jobname": os.getenv("SLURM_JOB_NAME"),
            "script_cfg": self._cfg,
            "world_size": get_world_size(),
        }
        self._send_requests(meta_data, msg_type="register", timeout=10)

    def send_keep_alive(self, step, loss, tgs, flops, ckpt_every):
        meta_data = {
            "jobid": os.getenv("SLURM_JOB_ID"),
            "rank": self._rank,
            "step": step,
            "loss": loss,
            "tgs": tgs,
            "flops": flops,
            "ckpt_every": ckpt_every,
        }
        self._send_requests(meta_data, msg_type="keep_alive")

    def send_exception(self, exception_msg: str):
        meta_data = {
            "jobid": os.getenv("SLURM_JOB_ID"),
            "rank": self._rank,
            "exception_msg": {"error": str(exception_msg)},
        }
        self._send_requests(meta_data, msg_type="catch_exception")


def init_local_adapter(global_rank, local_rank, cfg, coordinator_ip=None, coordinator_port=None):
    global local_adapter
    local_adapter = Adapter(
        rank=global_rank,
        local_rank=local_rank,
        cfg=cfg,
        coordinator_ip=coordinator_ip,
        coordinator_port=coordinator_port,
    )


def send_keep_alive(step, loss, tgs, tflops, ckpt_every):
    cur_loss = loss
    if isinstance(cur_loss, torch.Tensor):
        cur_loss = cur_loss.tolist()[0]
    if local_adapter:
        local_adapter.send_keep_alive(step, cur_loss, tgs, tflops, ckpt_every)


def send_exception(except_msg: str):
    if local_adapter:
        local_adapter.send_exception(except_msg)
