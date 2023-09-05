import logging
import os
import socket
import sys
import traceback
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from internlm.core.context import global_context as gpc


def tb_save_run_info(writer, config_lines, global_step=0):
    writer.add_text(tag="cmd", text_string=" ".join(sys.argv[:]), global_step=global_step)
    lines = []
    for line in config_lines:
        if line.strip().startswith("#"):
            continue
        lines.append(line)
    writer.add_text(tag="config", text_string="\n".join(lines), global_step=global_step)


def init_tb_writer(
    job_name: str,
    launch_time: str,
    file_name: str,
    tensorboard_folder: str,
    resume_tb_folder: str,
    step_count: int,
    config: str,
    logger: logging.Logger,
):
    tb_log_file_name = file_name
    if not tensorboard_folder:
        tb_folder = os.path.join(job_name, launch_time, "tensorboards")
    else:
        tb_folder = tensorboard_folder

    if gpc.get_global_rank() == 0:
        # If we don't load ckpt, 'resume_tb_folder' is set as the tensorboard
        # dir of the last task by 'make_launch_script.sh'.
        # If we load ckpt, 'resume_tb_folder' will be overwritten as the
        # reloaded 'train_state.resume_tb_folder'.s
        if resume_tb_folder is not None:
            assert len(resume_tb_folder) > 0 and resume_tb_folder != "/"
            if not os.path.exists(resume_tb_folder):
                logger.error(
                    f"Can't found resume_tb_folder{resume_tb_folder}, \
please make sure this folder is located at local file system."
                )
            else:
                logger.info(f"Try mv tensorboard logs: {resume_tb_folder} to {tb_folder}... ")
                os.system(f"cp -r {resume_tb_folder}/* {tb_folder}/")
                os.system(f"chmod -R +w {tb_folder}/")
        else:
            logger.info(f"Login tensorboard logs to: {tb_folder}")

        tb_logdir = os.path.join(tb_folder, tb_log_file_name)
        writer = SummaryWriter(log_dir=tb_logdir, max_queue=5, purge_step=step_count, flush_secs=3)
        writer.add_text(tag="job_name", text_string=job_name, global_step=step_count)
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
        job_name (str): The name of training job, defaults to None.
        launch_time (str): A string representing the launch time of the training.
        file_name (str): The log file name, defaults to None.
        tensorboard_folder (str): A string representing the folder for saving tensorboard logs.
        resume_tb_folder (str): A string representing the folder for resuming tensorboard logs.
        step_count (int): An integer representing the step count of the training.
        config (str): A string representing the configuration of the training.
        logger (logging.Logger): A logging.Logger object for logging information during training.
        enable_tb (bool): A boolean indicating whether to enable the tensorboard writer.

    """

    def __init__(
        self,
        job_name: str = None,
        launch_time: str = None,
        file_name: str = None,
        tensorboard_folder: str = None,
        resume_tb_folder: str = None,
        step_count: int = 0,
        config: str = None,
        logger: logging.Logger = None,
        enable_tb: bool = True,
    ) -> None:
        self.enable_tb = enable_tb
        self.tb_writer, self.tb_logdir = init_tb_writer(
            job_name=job_name,
            launch_time=launch_time,
            file_name=file_name,
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

    def add_scalars(self, key, value, step):
        try:
            assert isinstance(value, dict)
            if self.enable_tb and self.tb_writer is not None:
                self.tb_writer.add_scalars(main_tag=key, tag_scalar_dict=value, global_step=step)
        except Exception:
            traceback.print_exc()

    def add_text(self, key, value, step):
        try:
            if self.enable_tb and self.tb_writer is not None:
                self.tb_writer.add_text(tag=key, text_string=value, global_step=step)
        except Exception:
            traceback.print_exc()
