import os
import torch
import socket
import functools
import contextlib
from internlm.utils.registry import MODEL_INITIALIZER
from internlm.utils.writer import Writer
from internlm.utils.megatron_timers import megatron_timer
from internlm.utils.megatron_timers import megatron_timer as timer
from typing import Callable, Iterable, Union
from internlm.utils.logger import get_logger, initialize_uniscale_logger
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.parallel import get_parallel_log_file_name
from internlm.utils.common import DummyProfile
from internlm.utils.gputest import bench_gpu, bench_net
from beautifultable import BeautifulTable
from collections import defaultdict
import torch.distributed as dist
import configs.profiler as prof_config


logger = get_logger(__file__)

class LLMManager:
    def __init__(
        self,
        timer: megatron_timer,
        train_state,
        log_time_step: int = 1,
        do_trace_profiling: bool = False,
        trace_profiling_range: Callable = None,
        writer: Writer = None,
        current_time = None,
        memory_profiler = None,
    ) -> None:
        """
        TODO: support PP diagnosis mode.

        Args:
            timer (megatron_timer): 
            train_state (_type_): 
            log_time_step (int, optional): . Defaults to 1.
            trace_profiling_range (Callable, optional): A user-defined function with a return value of true or false, 
                used to control whether trace profiling is enabled. Defaults to None.
            launch_time (str, optional): . Defaults to None.
            active_count (int, optional): trace profiling interval. Defaults to 1.
            do_trace_profiling (bool): Whether to enable trace profiling. Defaults to False.
            do_diagnosis (bool, optional): Whether to enable runtime diagnostics. Defaults to False.
            writer (Writer, optional): . Defaults to None.
        """
        from datetime import datetime

        self.trace_profiling = do_trace_profiling
        self.train_state = train_state
        self.log_time_step = log_time_step
        self.writer = writer
        self.timer: megatron_timer = timer
        self.current_time = current_time
        self.torch_profile = DummyProfile()
        self.rank_avg_time = defaultdict(int)
        self.memory_profiler = memory_profiler
        self.slower_last = 0
        
        default_configs = {
            "torch_active_count": None,
            "memory_active_count": None,
            "bench_active_count": None, 
            "diagnosis_active_count": None,
            "diagnosis_start": 10,
            "diagnosis_slower_check": 50
        }
        
        for key in default_configs:
            if key not in prof_config.profiler:
                setattr(self, key, default_configs[key])
            else:
                setattr(self, key, prof_config.profiler[key])
        
        # runtime time metrics.
        self.time_ckpts = [
            "batch-gen",
            "fwd",
            "bwd",
            "fwd-bwd",
            "dp_sync",
            "post_fn",
            "cal_loss",
            "sync_grad",
            "cal_norm",
            "step",
            "one-batch",
        ]

        if trace_profiling_range is None:
            trace_profiling_range = (
                lambda: gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0
            )

        if self.trace_profiling:
            if trace_profiling_range():
                self.torch_profile = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(skip_first=5, wait=1, warmup=1, active=2, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        f"{gpc.config.JOB_NAME}/{current_time}/traces/",
                    ),
                    with_stack=True,
                    with_modules=True,
                )
            else:
                self.torch_profile = DummyProfile()
        else:
            self.torch_profile = DummyProfile()
        
        if self.diagnosis_active_count:
            log_file_name = 'diagnosis.log'
            log_folder = os.path.join(gpc.config.JOB_NAME, self.current_time, "logs")
            log_dir = os.path.join(log_folder, log_file_name)
            self.diagnosis_path = log_dir
            
    def get_rank_uid(self):
        return (
            f"{socket.gethostname()}_rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"pp{gpc.get_local_rank(ParallelMode.PIPELINE)}"
        )

    def __enter__(self):
        # self.torch_profile.__enter__()
        return self
    
    def _print_table(self, all_times):
        table = BeautifulTable(maxwidth=200)
        avg_vals_dict = defaultdict(int)
        for key, values in all_times.items():
            all_times[key] = sorted(values, key=lambda x: x[0], reverse=True)
            avg_value = round(sum(x[0] for x in all_times[key][1:-1]) / (len(all_times[key]) - 2), 2)
            avg_vals_dict[key] = avg_value
            
            subtable_order = BeautifulTable()
            subtable_ruid = BeautifulTable()
            subtable_value = BeautifulTable()
            
            order = 1
            for value, ruid in all_times[key]:
                if value > avg_value * 1.05:
                    subtable_order.rows.append([order])
                    subtable_ruid.rows.append([ruid])
                    subtable_value.rows.append([value])
                    order += 1
                    
            subtable_order.border.left = ''
            subtable_order.border.right = ''
            subtable_order.border.top = ''
            subtable_order.border.bottom = ''
            
            subtable_ruid.border.left = ''
            subtable_ruid.border.right = ''
            subtable_ruid.border.top = ''
            subtable_ruid.border.bottom = ''

            subtable_value.border.left = ''
            subtable_value.border.right = ''
            subtable_value.border.top = ''
            subtable_value.border.bottom = ''
            
            row_header = key + '\n' + f'avg:{avg_value}'
            table.rows.append([row_header, subtable_order, subtable_ruid, subtable_value])
        
        table.columns.header = ['', 'order', 'ruid', 'value']
        
        print('batch_count', self.train_state.batch_count)
        print("Inner-step dia:")
        print(table)
        print()
        
    def step(self):
        batch_count = self.train_state.batch_count
        
        if self.memory_active_count and self.memory_profiler is not None:
            if batch_count % self.memory_active_count == 0:
                self.memory_profiler.step()
        # Try increase torch trace profiler counter
        if self.torch_active_count:
            if batch_count % self.torch_active_count == 0:
                self.torch_profile.step()

        # Try dump timer to rb or log.
        if (self.train_state.step_count + 1) % self.log_time_step == 0:
            if self.writer:
                self.timer.write(
                    self.time_ckpts,
                    self.writer,
                    self.train_state.step_count,
                    normalizer=self.log_time_step,
                    reset=False,
                )
            if gpc.is_rank_for_log():
                self.timer.log(
                    names=self.time_ckpts, logger=logger, normalizer=self.log_time_step, reset=False
                )
                 
        # If we are in diagnosis mode, rank 0 will gahter all rank runtime time info.
        if self.diagnosis_active_count and batch_count > self.diagnosis_start:
            global_rank_uid = self.get_rank_uid()
            time_info = self.timer.get_all_timer_results(reset=False)
            with open(self.diagnosis_path, 'a') as f:
                with contextlib.redirect_stdout(f):
                    if batch_count % self.diagnosis_active_count == 0:
                        time_list = (global_rank_uid, time_info)
                        all_rank_time_list = [None for _ in range(gpc.get_world_size(ParallelMode.GLOBAL))]
                        dist.all_gather_object(all_rank_time_list, time_list, group=gpc.get_group(ParallelMode.GLOBAL))
                    
                        if gpc.is_rank_for_log():
                            all_times = {}
                            for rank_time_info in all_rank_time_list:
                                ruid, info = rank_time_info
                                for time_tuple in info:
                                    name, value = time_tuple
                                    if name not in all_times:
                                        all_times[name] = [(value, ruid)]
                                    else:
                                        all_times[name].append((value, ruid))
                                                                    
                            self._print_table(all_times)
                                        
                    if gpc.is_rank_for_log(): 
                        slow_list = []        
                        for time_tuple in time_info:
                            name, value = time_tuple      
                            avg_val = self.rank_avg_time[name]
                            if batch_count > (self.diagnosis_start + 1) and value > avg_val * 1.05:
                                slow_list.append((name, value, avg_val))
                            sum_num = batch_count - self.diagnosis_start - 1
                            sum_val = self.rank_avg_time[name] * (sum_num) + value
                            self.rank_avg_time[name] = round(sum_val / (sum_num + 1), 2)
                            
                        if slow_list != []:
                            self.slower_last += 1
                            if self.slower_last == self.diagnosis_slower_check:
                                self.slower_last = 0
                                print(f"Warning: step:{batch_count}, The delay has continued to increase for {self.diagnosis_slower_check} steps")
                        else:
                            self.slower_last = 0
                                            
                        if batch_count % self.diagnosis_active_count == 0 and slow_list != []:
                            print('Cross step dia:')
                            print('This step is slower than average')
                            for tuple in slow_list:
                                name, value, avg_val = tuple
                                print(f'{name} = {value} > avg ({avg_val})')
                            print()
                            
        self.timer.reset()

        # Do cuda burn test


        # Do nccl-test benchmark
        if self.bench_active_count and batch_count % self.bench_active_count == 0:
            bench_gpu()
            bench_net()


    def __exit__(self, a, b, c):
        # self.torch_profile.__exit__(a, b, c)
        # return self
        pass
