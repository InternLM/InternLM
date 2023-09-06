import fcntl
import os
import time
from multiprocessing import Process

import pytest
import torch
import torch.distributed as dist

os.environ["INTERNLM_ENABLE_TIMEOUT"] = "1"  # noqa  # pylint: disable=wrong-import-position
os.environ["NCCL_TIMEOUT"] = "5"
from internlm.utils.timeout import llm_timeout
from tests.test_utils.common_fixture import (  # noqa # pylint: disable=unused-import
    init_config,
)

WORLD_SIZE = 2


@llm_timeout(2, "fake_timeout_func")
def fake_timeout_func():
    time.sleep(10)


@llm_timeout(10, "nccl_timeout_func")
def nccl_timeout_func(rank):
    # see: https://github.com/pytorch/pytorch/issues/104506#issuecomment-1679762880
    # 'NCCL_ASYNC_ERROR_HANDLING' cannot take effect on the first collective communication.
    buff = torch.ones([64, 64]).cuda(rank)
    dist.all_reduce(buff)  # lazy communicator init
    torch.cuda.synchronize()
    if rank == 0:
        dist.all_reduce(buff)
        torch.cuda.synchronize()  # main thread will hang at here.
    else:
        time.sleep(9999)


@llm_timeout(10, "try_file_lock")
def try_file_lock(rank, stop_file_path):
    if rank == 1:
        time.sleep(5)

    with open(stop_file_path, "r", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # rank 1 hang.
        if rank == 0:
            time.sleep(99999)  # rank 0 hang.
        f.seek(0)
        f.read()
        fcntl.flock(f, fcntl.LOCK_UN)


def local_timeout(rank, _):

    try:
        fake_timeout_func()
    except TimeoutError as e:
        print(f"local_timeout, rank:{rank}, e:{e}", flush=True)
    else:
        assert False, "It should timeout!"


def gpc_timeout(rank, world_size):

    from internlm.initialize import initialize_distributed_env

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12377"
    initialize_distributed_env(config=init_config, launcher="torch", master_port=12377, args_check=False)

    try:
        nccl_timeout_func(rank)
    except TimeoutError as e:
        print(f"gpc_timeout, rank:{rank}, e:{e}", flush=True)
        time.sleep(5)  # wait rank 0 to be killed
    else:
        time.sleep(5)  # give some time to let Watchdog kill rank 0.
        assert False, "It should timeout!"


def file_lock_timeout(rank, _, stop_file_path):
    if rank == 0:
        with open(stop_file_path, "w"):
            pass
    try:
        try_file_lock(rank, stop_file_path)
    except TimeoutError as e:
        print(e, flush=True)
    else:
        assert False, "It should timeout!"
    finally:
        if rank == 0:
            os.remove(stop_file_path)


timeout_func_list = [(gpc_timeout, 2, None), (local_timeout, 1, None), (file_lock_timeout, 2, "test_lock.log")]


@pytest.mark.parametrize("timeout_func_and_args", timeout_func_list)
def test_timeout(timeout_func_and_args):
    timeout_func, world_size, other_args = timeout_func_and_args
    procs = []
    for i in range(world_size):
        if other_args is None:
            args = (i, world_size)
        else:
            args = (i, world_size, other_args)
        proc = Process(target=timeout_func, args=args)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join(15)
        if proc.is_alive():
            proc.terminate()
            proc.join()
