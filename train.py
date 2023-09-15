#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import socket
import time
import traceback
from functools import partial

import torch
import torch.distributed as dist

import internlm
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.scheduler import SchedulerMetricHook
from internlm.core.trainer import TrainState
from internlm.initialize import initialize_distributed_env
from internlm.model.loss import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex
from internlm.monitor import initialize_monitor_manager, send_alert_message
from internlm.monitor.monitor import monitor_manager as mm
from internlm.train import (
    get_train_data_loader,
    get_validation_data_loader,
    initialize_llm_profile,
    initialize_model,
    initialize_optimizer,
    load_new_batch,
    record_current_batch_training_metrics,
)
from internlm.utils.common import (
    BatchSkipper,
    get_megatron_flops,
    launch_time,
    parse_args,
)
from internlm.utils.evaluation import evaluate_on_val_dls
from internlm.utils.gputest import empty_cache_and_diag
from internlm.utils.logger import get_logger, initialize_uniscale_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.model_checkpoint import CheckpointManager
from internlm.utils.parallel import get_parallel_log_file_name
from internlm.utils.simple_memory_profiler import SimpleMemoryProfiler
from internlm.utils.writer import Writer

# global llm logger
logger = get_logger(__file__)


def initialize_llm_logger(start_time: str):
    """
    Initialize customed uniscale logger.

    Args:
        start_time (str): The launch time of current training job.

    Returns: The instance of uniscale logger.
    """

    uniscale_logger = initialize_uniscale_logger(
        job_name=gpc.config.JOB_NAME, launch_time=start_time, file_name=get_parallel_log_file_name()
    )
    if uniscale_logger is not None:
        global logger
        logger = uniscale_logger

    return uniscale_logger


def main(args):
    # init setting
    skip_batches = gpc.config.data.skip_batches
    total_steps = gpc.config.data.total_steps
    valid_every = gpc.config.data.valid_every
    label_smoothing = gpc.config.loss.label_smoothing

    get_tflops_func = partial(
        get_megatron_flops,
        checkpoint=gpc.config.model.checkpoint,
        seq_len=gpc.config.SEQ_LEN,
        hidden_size=gpc.config.model.hidden_size,
        num_layers=gpc.config.model.num_layers,
        vocab_size=gpc.config.model.vocab_size,
        global_batch_size=gpc.config.data.micro_bsz * gpc.config.data.micro_num * gpc.get_world_size(ParallelMode.DATA),
        global_world_size=gpc.get_world_size(ParallelMode.GLOBAL),
        mlp_ratio=gpc.config.MLP_RATIO,
    )

    # get and broadcast current time
    current_time = launch_time()
    objs = [current_time]
    dist.broadcast_object_list(objs, src=0)
    current_time = objs[0]

    # initialize customed llm logger
    uniscale_logger = initialize_llm_logger(start_time=current_time)

    # initialize model
    model = initialize_model()

    with open(args.config, "r") as f:
        config_lines = f.readlines()

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=label_smoothing)

    # initialize the train and validation data loader
    train_dl, dataset_types = get_train_data_loader(num_worker=4)
    val_dls = get_validation_data_loader()

    # initialize and resume train state
    train_state = TrainState(gpc.config, train_dl.batch_sampler)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=model)

    ckpt_manager = CheckpointManager(
        ckpt_config=gpc.config.ckpt,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dl=train_dl,
        model_config=gpc.config.model,
        model_config_file="".join(config_lines),
        feishu_address=gpc.config.monitor.alert.feishu_alert_address,
    )

    # Loading other persistent training states.
    ckpt_manager.try_resume_training(train_state, current_time)

    # initialize customed llm writer
    writer = Writer(
        job_name=gpc.config.JOB_NAME,
        launch_time=current_time,
        file_name=get_parallel_log_file_name(),
        tensorboard_folder=gpc.config.tensorboard_folder,
        resume_tb_folder=train_state.resume_tb_folder,  # resume from ckpt.
        step_count=train_state.step_count,  # resume from ckpt.
        config=config_lines,
        logger=logger,
        enable_tb=gpc.config.enable_tb,
    )

    # initialize metric for calculating accuracy and perplexity
    metric = AccPerplex(
        device=torch.cuda.current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
        dataset_types=dataset_types,
    )

    # initialize trainer
    scheduler_hooks = [
        SchedulerMetricHook(
            metric=metric,
            skip=(
                gpc.is_using_pp()
                and hasattr(gpc.config.model, "num_chunks")
                and gpc.config.model.num_chunks > 1
                and gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
            ),
        ),
    ]

    trainer, train_dl, _, _ = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dl,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=scheduler_hooks,
    )

    # initialize simple memory profiler
    if args.profiling:
        memory_profiler = SimpleMemoryProfiler(
            model,
            optimizer.optim,
            log_folder=f"memory_trace/rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}",
        )
    else:
        memory_profiler = None

    # initialize the batch skipper
    batch_skipper = BatchSkipper(skip_batches)

    trainer.train()

    # transfer the train data loader into train data iterator
    train_iter = iter(train_dl)

    with initialize_llm_profile(profiling=args.profiling, start_time=current_time) as prof:
        # start iterating the train data and begin training
        for batch_count in range(train_state.batch_count, total_steps):
            empty_cache_and_diag(batch_count, interval=gpc.config.data.empty_cache_and_diag_interval)
            start_time = time.time()
            timer("one-batch").start()

            # load batch data
            batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter, train_state=train_state)

            # record the consumed samples in training
            train_state.batch_count = batch_count
            train_state.num_consumed_samples_in_epoch += len(batch[1])
            if batch_skipper(batch_count):  # skip this batch
                if gpc.is_rank_for_log():
                    logger.info(f"Skip batch count:`{batch_count}`...")
                timer("one-batch").stop()
                continue

            # zero the grads of parameters
            trainer.zero_grad()
            # process data
            if batch[0].get("type_ids", None) is not None:
                metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))

            # do forward and backward
            timer("fwd-bwd").start()

            _, _, loss = trainer.execute_schedule(
                batch, forward_only=False, return_loss=True, return_output_label=False
            )
            timer("fwd-bwd").stop()

            # update parameters, and returns (success_update, grad_norm)
            trainer_result = trainer.step()
            assert trainer_result is not None

            success_update, grad_norm_groups = trainer_result
            if success_update:  # update parameters successfully
                train_state.step_count += 1
            else:
                train_state.inf_nan_skip_batches += 1  # record the amount of updating parameters unsuccessfully.
                if -1 in grad_norm_groups.values() and gpc.is_rank_for_log():  # -1 encodes a specific failure case
                    logger.warning(f"Warning: skip parameter update at step {batch_count}.")
                    send_alert_message(
                        address=gpc.config.monitor.alert.feishu_alert_address,
                        message=f"Warning: skip parameter update at step {batch_count}.",
                    )

            # calculate and record the training metrics, eg. loss, accuracy and so on.
            record_current_batch_training_metrics(
                get_tflops_func=get_tflops_func,
                logger=logger,
                writer=writer,
                success_update=success_update,
                batch_count=batch_count,
                batch=batch,
                train_state=train_state,
                optimizer=optimizer,
                beta2_scheduler=beta2_scheduler,
                trainer=trainer,
                start_time=start_time,
                loss=loss,
                grad_norm=grad_norm_groups,
                metric=metric,
                update_panel=uniscale_logger is not None,
            )

            timer("one-batch").stop()

            # evaluate on validation data loaders
            if valid_every > 0 and train_state.step_count % valid_every == 0:
                evaluate_on_val_dls(
                    trainer=trainer,
                    val_dls=val_dls,
                    writer=writer,
                    logger=logger,
                    step_count=train_state.step_count,
                    update_panel=uniscale_logger is not None,
                )

            # checkpoint the training states in specific steps, which is determined by the args "checkpoint_every"
            # # save batch sampler that tracks the true consumed samples
            now_break = ckpt_manager.try_save_checkpoint(train_state)
            if now_break:
                break

            if memory_profiler is not None:
                memory_profiler.step()

            if batch_count % 2 == 0:
                prof.step()

    ckpt_manager.wait_async_upload_finish()


if __name__ == "__main__":
    args = parse_args()
    hostname = socket.gethostname()

    # initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # initialize monitor manager context
    with initialize_monitor_manager(
        job_name=gpc.config.JOB_NAME, alert_address=gpc.config.monitor.alert.feishu_alert_address
    ):
        try:
            main(args)
        except Exception:
            logger.error(
                f"Raise exception from {hostname} with rank id: {gpc.get_global_rank()}\n{traceback.format_exc()}",
            )
            mm.monitor_exception(
                alert_address=gpc.config.monitor.alert.feishu_alert_address, excp_info=traceback.format_exc()
            )
