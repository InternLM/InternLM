#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import socket

import torch
import torch.distributed as dist
from flash_attn.modules.mha import FlashSelfAttention, SelfAttention
from torch.utils import benchmark

from internlm.monitor import send_alert_message
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer

try:
    import GPUtil
    import psutil
except ImportError:
    GPUtil, psutil = None, None

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.common import get_current_device

logger = get_logger(__file__)


def empty_cache_and_diag(batch_count, interval=50):
    """empty cuda cache and run diag bench or tests."""
    if interval <= 0:
        interval = 50
    if batch_count % int(interval) == 0:
        # there is no need to do diag on the first batch
        if batch_count > 0:
            if gpc.is_rank_for_log():
                logger.info("Empty Cache and Diagnosis GPU/NCCL/Timer ...")
            with torch.no_grad():
                timer_diagnosis()
                bench_gpu()
                bench_net()
        # do empty_cache after the bench
        torch.cuda.empty_cache()


def benchmark_forward(
    test_fn,
    *inputs,
    repeats=100,
    amp=True,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            test_fn(*inputs, **kwinputs)

    bench_timer = benchmark.Timer(
        stmt="test_fn_amp(*inputs, **kwinputs)",
        globals={"test_fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    used_time = bench_timer.timeit(repeats)
    return used_time.mean


def flops(batch, seqlen, headdim, nheads, time_f):
    """Compute the flops value of a GPU with give flashattention function"""

    flop = 4 * batch * seqlen**2 * nheads * headdim
    return (flop / time_f / 10**12) if not math.isnan(time_f) else 0.0


def get_gpu_temperature():
    """Get current GPU temperature."""
    try:
        gpu_id = torch.cuda.current_device()
    except AssertionError:
        gpu_id = -1

    if GPUtil is not None and gpu_id >= 0:
        gpus = GPUtil.getGPUs()
        gpu_temperature = gpus[gpu_id].temperature
    else:
        gpu_temperature = -1

    return gpu_temperature


def get_cpu_temperature():
    """Get current CPU temperature."""

    if psutil is not None:
        cpu_temperature = psutil.sensors_temperatures()["coretemp"][0].current
    else:
        cpu_temperature = -1

    return cpu_temperature


def timer_diagnosis():
    """Diagnosis running time"""

    if len(timer.names) == 0 or len(timer.times) == 0:
        return

    world_size = gpc.get_world_size(ParallelMode.DATA)
    if world_size < 2:
        return

    # if gpc.is_rank_for_log():
    #     logger.info("Diagnosis running timers ...")

    # detect slow rank compared to other ranks in the same DP group
    running_time = torch.Tensor(timer.times).to(device=get_current_device())
    avg_time = running_time.detach().clone()
    if world_size <= 4:
        dist.all_reduce(avg_time, op=torch.distributed.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))
    else:
        running_time_max = avg_time.detach().clone()
        running_time_min = avg_time.detach().clone()
        dist.all_reduce(running_time_max, op=torch.distributed.ReduceOp.MAX, group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(running_time_min, op=torch.distributed.ReduceOp.MIN, group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(avg_time, op=torch.distributed.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))
        avg_time = (avg_time - running_time_max - running_time_min) / (world_size - 2)

    diag_result = running_time > avg_time * gpc.config.data.diag_outlier_ratio
    diag_result = diag_result.tolist()
    avg_time = avg_time.tolist()

    for slow, name, time, avg in zip(diag_result, timer.names, timer.times, avg_time):
        if slow is False or avg < 0.5:
            continue
        msg = (
            f"Rank {gpc.get_local_rank(ParallelMode.GLOBAL)} is slower than avg on {name}, "
            f"Hostname {socket.gethostname()}, "
            f"its time {time:.2f}, avg {avg:.2f}, "
            f"CPU temp {get_cpu_temperature()}, GPU temp { get_gpu_temperature()}"
        )
        logger.warning(msg)
        send_alert_message(
            address=gpc.config.monitor.alert.feishu_alert_address,
            message=msg,
        )

    # detect slow rank compared to historical timer data
    for name, time in zip(timer.names, timer.times):
        if name not in timer.hist or len(timer.hist[name]) < 5:
            continue
        hist_avg = sum(timer.hist[name]) / len(timer.hist[name])
        if time > hist_avg * gpc.config.data.diag_outlier_ratio and time > 0.5:
            msg = (
                f"Rank {gpc.get_local_rank(ParallelMode.GLOBAL)} is slower than hist avg on {name}, "
                f"Hostname {socket.gethostname()}, "
                f"its time {time:.2f}, hist_avg {hist_avg:.2f}, "
                f"CPU temp {get_cpu_temperature()}, GPU temp { get_gpu_temperature()}"
            )
            logger.warning(msg)
            send_alert_message(
                address=gpc.config.monitor.alert.feishu_alert_address,
                message=msg,
            )


def bench_net():
    """Benchmark nccl performance for slow node detection."""

    if gpc.get_world_size(ParallelMode.GLOBAL) <= 1:
        return

    # if gpc.is_rank_for_log():
    #     logger.info("benchmarking network speed ...")

    repeats = 100
    input_data = torch.randn(
        8 * 1024 * 1024,
        device=get_current_device(),
        dtype=torch.bfloat16,
    )

    def allreduce_fn(inputs):
        dist.all_reduce(inputs, op=torch.distributed.ReduceOp.AVG, group=gpc.get_group(ParallelMode.NETTEST))

    bench_timer = benchmark.Timer(
        stmt="test_fn_amp(inputs)",
        globals={"test_fn_amp": allreduce_fn, "inputs": input_data},
        num_threads=torch.get_num_threads(),
    )
    allreduce_time = bench_timer.timeit(repeats).mean
    allreduce_time = allreduce_time * 10**3
    allreduce_time_this = allreduce_time
    allreduce_time = torch.Tensor([allreduce_time]).to(device=get_current_device())
    dist.all_reduce(allreduce_time, group=gpc.get_group(ParallelMode.GLOBAL))
    allreduce_time_avg = allreduce_time / gpc.get_world_size(ParallelMode.GLOBAL)
    allreduce_time_avg = float(allreduce_time_avg.item())

    if allreduce_time_this >= allreduce_time_avg * gpc.config.data.diag_outlier_ratio:
        msg = (
            f"Rank {gpc.get_local_rank(ParallelMode.GLOBAL)} NCCL test is slower than avg, "
            f"Hostname {socket.gethostname()}, "
            f"allreduce_time {allreduce_time_this:.2f}, avg {allreduce_time_avg:.2f}, "
            f"CPU temp {get_cpu_temperature()}, GPU temp { get_gpu_temperature()}"
        )
        logger.warning(msg)
        send_alert_message(
            address=gpc.config.monitor.alert.feishu_alert_address,
            message=msg,
        )


def bench_gpu(use_flash_attn=True):
    """Benchmark single GPU performance for slow node detection."""

    # if gpc.is_rank_for_log():
    #     logger.info("benchmarking gpu speed ...")

    headdim = 64
    dim = 2048
    batch_size, seqlen = 2, 1024
    nheads = dim // headdim

    inner_attn = FlashSelfAttention if use_flash_attn else SelfAttention
    inner_attn = inner_attn(causal=True, softmax_scale=None, attention_dropout=0)

    qkv = torch.randn(
        batch_size,
        seqlen,
        3,
        dim // headdim,
        headdim,
        device=get_current_device(),
        dtype=torch.float16,
        requires_grad=True,
    )
    time_f = benchmark_forward(inner_attn, qkv)
    speed = flops(batch_size, seqlen, headdim, nheads, time_f)
    speed_this = speed
    speed = torch.Tensor([speed]).to(device=get_current_device())
    dist.all_reduce(speed, group=gpc.get_group(ParallelMode.GLOBAL))
    speed_avg = speed / gpc.get_world_size(ParallelMode.GLOBAL)
    speed_avg = float(speed_avg.item())

    if speed_this <= speed_avg / gpc.config.data.diag_outlier_ratio:
        msg = (
            f"Rank {gpc.get_local_rank(ParallelMode.GLOBAL)} GPU is slower than avg, "
            f"Hostname {socket.gethostname()}, "
            f"tflops {speed_this:.2f}, avg {speed_avg:.2f}, "
            f"CPU temp {get_cpu_temperature()}, GPU temp { get_gpu_temperature()}"
        )
        logger.warning(msg)
        send_alert_message(
            address=gpc.config.monitor.alert.feishu_alert_address,
            message=msg,
        )
