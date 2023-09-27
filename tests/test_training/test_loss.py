import math
import os
import subprocess

import pytest
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
from internlm.train import (
    get_train_data_loader,
    initialize_model,
    initialize_optimizer,
    load_new_batch,
)
from internlm.utils.common import BatchSkipper, launch_time
from internlm.utils.gputest import empty_cache_and_diag
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.model_checkpoint import CheckpointManager

CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "./configs/7B_sft.py")
TOTAL_STEPS = 10
LOSS_SPIKE_LIMIT = 1.5
LOSS_DEVIATION_LIMIT = 0.2
BASELINE_LOSS_LIST = [
    11.64188003540039,
    7.9205322265625,
    6.944362163543701,
    6.147305488586426,
    6.060564994812012,
    5.660439491271973,
    5.19430685043335,
    5.157323837280273,
    4.769168376922607,
    4.449280738830566,
]
cur_loss_list = []


def train(
    dp_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    num_chunks: int = 2,
    interleaved: bool = False,
    enable_sp: bool = False,
):
    # initialize distributed environment
    initialize_distributed_env(config=CONFIG_FILE_PATH)
    assert hasattr(gpc, "config") and gpc.config is not None

    # check parallel config
    assert (
        gpc.get_world_size(ParallelMode.DATA) == dp_size
    ), f"data parallel size: {gpc.get_world_size(ParallelMode.DATA)} is not as expected {dp_size}"
    assert (
        gpc.get_world_size(ParallelMode.TENSOR) == tp_size
    ), f"tensor parallel size: {gpc.get_world_size(ParallelMode.TENSOR)} is not as expected {tp_size}"
    assert (
        gpc.get_world_size(ParallelMode.PIPELINE) == pp_size
    ), f"pipeline parallel size: {gpc.get_world_size(ParallelMode.PIPELINE)} is not as expected {pp_size}"
    if interleaved:
        assert (
            gpc.is_using_pp() and hasattr(gpc.config.model, "num_chunks") and gpc.config.model.num_chunks == num_chunks
        )
        assert gpc.config.parallel["pipeline"].get(
            "interleaved_overlap", False
        ), "interleaved overlap must be enabled when using interleave pipeline scheduler"
    if enable_sp:
        assert gpc.config.parallel.get(
            "sequence_parallel", False
        ), "sequence_parallel must be True when enable_sp is True"

    # init setting
    gpc.config.data.total_steps = TOTAL_STEPS
    gpc.config.lr_scheduler.total_steps = TOTAL_STEPS
    total_steps = gpc.config.data.total_steps
    skip_batches = gpc.config.data.skip_batches
    label_smoothing = gpc.config.loss.label_smoothing

    # get and broadcast current time
    current_time = launch_time()
    objs = [current_time]
    dist.broadcast_object_list(objs, src=0)
    current_time = objs[0]

    # initialize model
    model = initialize_model()

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=label_smoothing)

    # initialize the train data loader
    train_dl, dataset_types = get_train_data_loader(num_worker=4)

    # initialize and resume train state
    train_state = TrainState(gpc.config, train_dl.batch_sampler)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=model)

    with open(CONFIG_FILE_PATH, "r") as f:
        config_lines = f.readlines()
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

    # initialize the batch skipper
    batch_skipper = BatchSkipper(skip_batches)

    trainer.train()

    # transfer the train data loader into train data iterator
    train_iter = iter(train_dl)

    # start iterating the train data and begin training
    for batch_count in range(train_state.batch_count, total_steps):
        empty_cache_and_diag(batch_count, interval=gpc.config.data.empty_cache_and_diag_interval)
        timer("one-batch").start()

        # load batch data
        batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter, train_state=train_state)

        # record the consumed samples in training
        train_state.batch_count = batch_count
        train_state.num_consumed_samples_in_epoch += len(batch[1])
        if batch_skipper(batch_count):  # skip this batch
            if gpc.is_rank_for_log():
                print(f"Skip batch count:`{batch_count}`...")
            timer("one-batch").stop()
            continue

        # zero the grads of parameters
        trainer.zero_grad()
        # process data
        if batch[0].get("type_ids", None) is not None:
            metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))

        # do forward and backward
        timer("fwd-bwd").start()

        # Compatible for non-moe
        moe_loss = None
        if hasattr(gpc.config.model, "num_experts"):
            _, _, loss, moe_loss = trainer.execute_schedule(
                batch, forward_only=False, return_loss=True, return_output_label=False
            )
        else:
            _, _, loss = trainer.execute_schedule(
                batch, forward_only=False, return_loss=True, return_output_label=False
            )
        if gpc.is_rank_for_log():
            assert loss is not None and not math.isnan(loss.item())
            global cur_loss_list
            cur_loss_list.append((loss.item() - moe_loss.item() if moe_loss is not None else loss.item()))
        timer("fwd-bwd").stop()

        # update parameters, and returns (success_update, grad_norm)
        trainer_result = trainer.step()
        assert trainer_result is not None

        success_update, _ = trainer_result
        assert success_update, "Error: grad norm inf or nan occurs!"
        if success_update:  # update parameters successfully
            train_state.step_count += 1
        else:
            train_state.inf_nan_skip_batches += 1  # record the amount of updating parameters unsuccessfully.

        timer("one-batch").stop()


def check_loss_spike():
    if gpc.is_rank_for_log():
        for step in range(1, TOTAL_STEPS):
            assert (
                cur_loss_list[step] < cur_loss_list[step - 1] * LOSS_SPIKE_LIMIT
            ), f"The loss spike occurs, {cur_loss_list[step - 1]}->{cur_loss_list[step]}, please check it!"


def check_loss_accuracy():
    if gpc.is_rank_for_log():
        for cur, target in zip(cur_loss_list, BASELINE_LOSS_LIST):
            assert (
                abs(cur - target) < LOSS_DEVIATION_LIMIT
            ), f"The loss accuracy is abnormal, {target}->{cur}, please check it!"


@pytest.mark.training_8GPU
def test_training_loss_with_dp8():
    # model training
    train(dp_size=8)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_16GPU_8DP2TP
def test_training_loss_with_dp8_tp2():
    # model training
    train(dp_size=8, tp_size=2)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_16GPU_8DP2TPSP
def test_training_loss_with_dp8_tp2_sp():
    # model training
    train(dp_size=8, tp_size=2, enable_sp=True)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_16GPU_8DP2PP
def test_training_loss_with_dp8_pp2():
    # model training
    train(dp_size=8, pp_size=2)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_16GPU_8DP2PP_InterleavedOverlap
def test_training_loss_with_dp8_pp2_interleaved_overlap():
    # model training
    train(dp_size=8, pp_size=2, interleaved=True)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()
