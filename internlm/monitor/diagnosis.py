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


logger = get_logger(__file__)

# def initialize_llm_logger(start_time: str, do_diagnosis=False):
#     """
#     Initialize customed uniscale logger.

#     Args:
#         start_time (str): The launch time of current training job.

#     Returns: The instance of uniscale logger.
#     """
#     print('do_diagnosis', do_diagnosis, flush=True)
#     uniscale_logger = initialize_uniscale_logger(
#         job_name=gpc.config.JOB_NAME, launch_time=start_time, file_name=get_parallel_log_file_name(), do_diagnosis=do_diagnosis
#     )
#     # if uniscale_logger is not None:
#     #     global logger
#     #     logger = uniscale_logger

#     return uniscale_logger

class LLMProfiler:
    def __init__(
        self,
        timer: megatron_timer,
        train_state,
        log_time_step: int = 1,
        launch_time: str = None,
        active_count: int = 1,
        do_trace_profiling: bool = False,
        trace_profiling_range: Callable = None,
        do_diagnosis: bool = False,
        writer: Writer = None,
        current_time = None,
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
        self.active_count = active_count
        self.in_diagnosis = do_diagnosis
        self.diagnosis_path = None
        self.log_time_step = log_time_step
        self.writer = writer
        self.timer: megatron_timer = timer
        self.current_time = current_time
        # self.uniscale_logger = initialize_llm_logger(start_time=current_time, do_diagnosis=do_diagnosis)
        self.torch_profile = DummyProfile()
        # self.avg_vals_dict = defaultdict(lambda: (0, 0))
        self.rank_avg_time = defaultdict(int)
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

        if launch_time is None:
            launch_time = datetime.now().strftime("%H:%M:%S")

        if self.trace_profiling:
            if trace_profiling_range():
                self.torch_profile = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(skip_first=30, wait=1, warmup=1, active=1, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        f"{gpc.config.JOB_NAME}/{launch_time}/traces/",
                    ),
                    with_stack=True,
                    with_modules=True,
                )
            else:
                self.torch_profile = DummyProfile()
        else:
            self.torch_profile = DummyProfile()
        
        if self.in_diagnosis:
            log_file_name = 'diagnosis.log'
            log_folder = os.path.join(gpc.config.JOB_NAME, self.current_time, "logs")
            log_dir = os.path.join(log_folder, log_file_name)
            file_path = log_dir
            self.diagnosis_path = file_path
            
    def get_rank_uid(self):
        return (
            f"{socket.gethostname()}_rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"pp{gpc.get_local_rank(ParallelMode.PIPELINE)}"
        )

    def __enter__(self):
        return self
    
    def _print_table(self, all_times):
        table = BeautifulTable(maxwidth=200)
        avg_vals_dict = defaultdict(int)
        for key, values in all_times.items():
            all_times[key] = sorted(values, key=lambda x: x[0], reverse=True)
            avg_value = round(sum(x[0] for x in all_times[key]) / len(all_times[key]), 2)
            avg_vals_dict[key] = avg_value
            
            subtable_order = BeautifulTable()
            subtable_ruid = BeautifulTable()
            subtable_value = BeautifulTable()
            
            order = 1
            for value, ruid in all_times[key]:
                if value >= avg_value * 1.05:
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
        # Try increase torch trace profiler counter
        if self.train_state.step_count % self.active_count == 0:
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
        if self.in_diagnosis:
            global_rank_uid = self.get_rank_uid()
            time_info = self.timer.get_all_timer_results(reset=False)
            batch_count = self.train_state.batch_count
            if batch_count > 10:
                with open(self.diagnosis_path, 'a') as f:
                    with contextlib.redirect_stdout(f):
                        if batch_count % 2 == 0:
                            time_list = (global_rank_uid, time_info)
                            if gpc.get_global_rank() == 0:
                                all_rank_time_list = [None for _ in range(gpc.get_world_size(ParallelMode.GLOBAL))]
                            else:
                                all_rank_time_list = None
                            dist.gather_object(time_list, all_rank_time_list, dst=0, group=gpc.get_group(ParallelMode.GLOBAL))
                            if gpc.get_global_rank() == 0:
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
                                            
                        if gpc.get_global_rank() == 0: 
                            slow_list = []        
                            for time_tuple in time_info:
                                name, value = time_tuple
                                if batch_count % 2 == 0:        
                                    avg_val = self.rank_avg_time[name]
                                    if value >= avg_val * 1:
                                        slow_list.append((name, value, avg_val))
                                sum_val = self.rank_avg_time[name] * (batch_count - 10) + value
                                self.rank_avg_time[name] = round(sum_val / (batch_count - 9), 2)
                                             
                            
                            if batch_count % 2 == 0 and slow_list != []:
                                print('Cross step dia:')
                                print('This step is slower than average')
                                for tuple in slow_list:
                                    name, value, avg_val = tuple
                                    print(f'{name} = {value} > avg ({avg_val})')
                                print()
                        
                        if batch_count % 2 == 0:
                            torch.cuda.empty_cache()
                            bench_gpu()
                            bench_net()   
                                     
                               
        self.timer.reset()

        # Do cuda burn test


        # Do nccl-test benchmark


    def __exit__(self, a, b, c):
        pass
