import logging
import os
import socket
import sys
import traceback
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


def get_tb_log_file_name():
    if gpc.is_rank_for_log():
        tb_prefix = "main_"  # Indicates a rank with more output information
    else:
        tb_prefix = ""

    tb_log_file_name = (
        f"{tb_prefix}dp={gpc.get_local_rank(ParallelMode.DATA)}_"
        f"tp={gpc.get_local_rank(ParallelMode.TENSOR)}_pp={gpc.get_local_rank(ParallelMode.PIPELINE)}"
    )
    return tb_log_file_name


def copy_ignore_folder(source_path, target_path):
    os.system(f"cp -r {source_path}/* {target_path}/")


def tb_save_run_info(writer, config_lines, global_step=0):
    writer.add_text(tag="cmd", text_string=" ".join(sys.argv[:]), global_step=global_step)
    lines = []
    for line in config_lines:
        if line.strip().startswith("#"):
            continue
        lines.append(line)
    writer.add_text(tag="config", text_string="\n".join(lines), global_step=global_step)


def init_tb_writer(
    launch_time,
    tensorboard_folder: str,
    resume_tb_folder: str,
    step_count: int,
    config: str,
    logger: logging.Logger,
):
    tb_log_file_name = get_tb_log_file_name()
    if not tensorboard_folder:
        tb_folder = os.path.join(gpc.config.JOB_NAME, launch_time)
    else:
        tb_folder = tensorboard_folder

    if gpc.get_global_rank() == 0:
        if resume_tb_folder is not None:
            logger.info(f"Try mv tensorboard logs: {resume_tb_folder} to {tb_folder}...")
            copy_ignore_folder(resume_tb_folder, tb_folder)
        else:
            logger.info(f"Login tensorboard logs to: {tb_folder}")

        tb_logdir = os.path.join(tb_folder, tb_log_file_name)
        writer = SummaryWriter(log_dir=tb_logdir, max_queue=5, purge_step=step_count, flush_secs=3)
        writer.add_text(tag="job_name", text_string=gpc.config.JOB_NAME, global_step=step_count)
        writer.add_text(tag="tensorboard_folder", text_string=tb_logdir, global_step=step_count)

        torch.distributed.broadcast_object_list([tb_folder], src=0)
    else:
        objects = [None]
        torch.distributed.broadcast_object_list(objects, src=0)
        tb_folder = objects[0]
        tb_logdir = os.path.join(tb_folder, tb_log_file_name)
        writer = SummaryWriter(log_dir=tb_logdir, max_queue=5, purge_step=step_count, flush_secs=3)

    if gpc.is_rank_for_log():
        tb_save_run_info(
            writer=writer,
            config_lines=config,
            global_step=step_count,
        )

    writer.add_text(
        tag=f"mapping_{tb_log_file_name}",
        text_string=f"file_path={tb_logdir} hostname={socket.gethostname()} device={torch.cuda.current_device()}",
        global_step=step_count,
    )
    writer.add_scaler = partial(writer.add_scalar, new_style=True)

    return writer, tb_logdir


class Writer:
    """
    Customed writer based on tensorboard for recording training metrics.

    Args:

    Return:
    """

    def __init__(
        self,
        launch_time,
        tensorboard_folder: str = None,
        resume_tb_folder: str = None,
        step_count: int = 0,
        config: str = None,
        logger: logging.Logger = None,
        enable_tb: bool = True,
    ) -> None:
        self.enable_tb = enable_tb
        self.tb_writer, self.tb_logdir = init_tb_writer(
            launch_time=launch_time,
            tensorboard_folder=tensorboard_folder,
            resume_tb_folder=resume_tb_folder,
            step_count=step_count,
            config=config,
            logger=logger,
        )

    def add_scalar(self, key, value, step):
        try:
            if self.enable_tb and self.tb_writer is not None:
                self.tb_writer.add_scalar(tag=key, scalar_value=value, global_step=step)
        except Exception:
            traceback.print_exc()

    def add_text(self, key, value, step):
        try:
            if self.enable_tb and self.tb_writer is not None:
                self.tb_writer.add_text(tag=key, text_string=value, global_step=step)
        except Exception:
            traceback.print_exc()
