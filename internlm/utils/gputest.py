#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import gc
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

import re
import traceback

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.common import get_current_device

original_all_reduce = dist.all_reduce

logger = get_logger(__file__)


# Gloabl cuda cache flush counter
n_caching_allocator_flushes = 0


def empty_cache_and_diag(batch_count, interval=50):
    """empty cuda cache and run diag bench or tests."""
    if interval <= 0:
        interval = 50

    cuda_memory_analyze(batch_count, batch_count % int(interval) == 0 or batch_count <= 5)

    if batch_count % int(interval) == 0:
        # there is no need to do diag on the first batch
        if batch_count > 0:
            if gpc.is_rank_for_log():
                logger.info("Empty Cache and Diagnosis GPU/NCCL/Timer ...")
            with torch.no_grad():
                timer_diagnosis()
                bench_gpu()
                # FIXME: Runtime benchmark diagnosis can easily cause the training process
                # to exit due to NCCL errors.
                # bench_net()
        # do empty_cache after the bench
        torch.cuda.empty_cache()
        # do garbage collection
        gc.collect()


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


"""
Useful utility functions migrated from deepseped.
"""


def warmup_process_group():
    # Prevent OOM from nccl communication.
    if dist.is_initialized():
        buffer = torch.ones([64]).cuda()
        if gpc.is_initialized(ParallelMode.DATA):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.DATA))
        if gpc.is_initialized(ParallelMode.TENSOR):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.TENSOR))
        if gpc.is_initialized(ParallelMode.PIPELINE):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.PIPELINE))
        if gpc.is_initialized(ParallelMode.ZERO1):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.ZERO1))
        if gpc.is_initialized(ParallelMode.MODEL):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.MODEL))
        if gpc.is_initialized(ParallelMode.ZERO3_DP):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.ZERO3_DP))
        if gpc.is_initialized(ParallelMode.EXPERT_DATA):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.EXPERT_DATA))
        if gpc.is_initialized(ParallelMode.EXPERT):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.EXPERT))

        dist.barrier()
        del buffer
        torch.cuda.empty_cache()


def cuda_memory_analyze(step=0, print_mm_suage=False):
    global n_caching_allocator_flushes
    torch.cuda.synchronize()

    g_rank = gpc.get_global_rank()
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)
    rank_id = f"Rank:{g_rank}-tp{tp_rank}-pp{pp_rank}-dp{dp_rank}"

    if print_mm_suage and gpc.get_local_rank(ParallelMode.DATA) == 0:
        logger.info(
            f"{rank_id}: Step {step}: "
            f"Allocated {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),4 )} GB, "
            f"Max_Allocated {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),4)} GB, "
            f"Reserved {round(torch.cuda.memory_reserved()/ (1024 * 1024 * 1024),4)} GB, "
            f"Max_Reserved {round(torch.cuda.max_memory_reserved()/ (1024 * 1024 * 1024),4)} GB "
        )

        torch.cuda.reset_peak_memory_stats()

    # warn user about caching allocator flushes
    memory_stats = torch.cuda.memory_stats()
    alloc_retries = memory_stats.get("num_alloc_retries")
    if alloc_retries is None:
        alloc_retries = 0
    if alloc_retries > n_caching_allocator_flushes:
        retry_count = alloc_retries - n_caching_allocator_flushes
        if gpc.get_global_rank() == 0:
            logger.warning(
                f"{rank_id}: pytorch allocator cache flushes {retry_count} times since last step."
                "this happens when there is high memory pressure and is detrimental to "
                "performance. if this is happening frequently consider adjusting "
                "settings to reduce memory consumption. If you are unable to "
                "make the cache flushes go away consider adding "
                "torch.cuda.empty_cache() calls in your training loop to ensure "
                "that all ranks flush their caches at the same time"
            )
        n_caching_allocator_flushes = alloc_retries


def get_traceback_list():
    pattern = r"file ([^,]+), line (\d+)"
    traceback_list = list(traceback.extract_stack())
    result = []
    for item in traceback_list:
        item = str(item)
        match = re.search(pattern, item)
        if match:
            file_path = match.group(1)
            line_number = match.group(2)
            result.append(f"{file_path}, line {line_number}")

    return result


def diag_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    import time

    if "diag_all_reduce" in gpc.config and gpc.config.diag_all_reduce.enable_diag:
        diag_config = gpc.config.diag_all_reduce
        start_wait = time.time()
        handle = original_all_reduce(tensor, op=op, group=group, async_op=async_op)
        dist.barrier()
        wait_time = (gpc.get_global_rank(), time.time() - start_wait)

        object_gather_list = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(object_gather_list, wait_time, group)
        if dist.get_rank(group) == 0:
            sort_list = sorted(object_gather_list, key=lambda x: x[1])
            times = [tup[1] for tup in sort_list]
            average_val = sum(times) / len(times)
            range_val = sort_list[-1][1] - sort_list[0][1]
            if average_val > diag_config.average_val or range_val > diag_config.range_val:
                skip_first = diag_config.skip_first
                skip_last = diag_config.skip_last
                traceback_list = get_traceback_list()[skip_first : -skip_last if skip_last > 0 else None]
                result = {
                    "rank_time_sort (global_rank, )": sort_list,
                    "traceback": traceback_list,
                }
                logger.warning(result)
    else:
        handle = original_all_reduce(tensor, op=op, group=group, async_op=async_op)

    return handle


dist.all_reduce = diag_all_reduce
