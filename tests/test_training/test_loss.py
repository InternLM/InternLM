import math
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

CONFIG_FILE_PATH = "./configs/7B_sft.py"
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


def train():
    # initialize distributed environment
    initialize_distributed_env(config=CONFIG_FILE_PATH)
    assert hasattr(gpc, "config") and gpc.config is not None

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

        _, _, loss = trainer.execute_schedule(batch, forward_only=False, return_loss=True, return_output_label=False)
        if gpc.is_rank_for_log():
            assert loss is not None and not math.isnan(loss.item())
            global cur_loss_list
            cur_loss_list.append(loss.item())
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


class TestCaseTrain8GPU:
    """
    Test cases for Model Training with 8 GPUs.
    Parallel Config:
        data parallel size = 8.
    """

    @staticmethod
    def setup_class():
        # model training
        train()

        # print loss value
        print(f"cur_loss_list: {cur_loss_list}", flush=True)

    @staticmethod
    @pytest.mark.training_8GPU
    def test_loss_spike_with_dp8():
        check_loss_spike()

    @staticmethod
    @pytest.mark.training_8GPU
    def test_loss_accuracy_with_dp8():
        check_loss_accuracy()


class TestCaseTrain16GPUWith8DP2TP:
    """
    Test cases for Model Training with 16 GPUs.
    Parallel Config:
        data parallel size = 8.
        tensor parallel size = 2.
    """

    @staticmethod
    def setup_class():
        # update config tensor parallel size
        command = f"sed -i 's/^.*tensor=.*/    tensor=2,/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)

        # model training
        train()

        # print loss value
        print(f"cur_loss_list: {cur_loss_list}", flush=True)

    @staticmethod
    @pytest.mark.training_16GPU_8DP2TP
    def test_loss_spike_with_dp8_tp2():
        check_loss_spike()

    @staticmethod
    @pytest.mark.training_16GPU_8DP2TP
    def test_loss_accuracy_with_dp8_tp2():
        check_loss_accuracy()


class TestCaseTrain16GPUWith8DP2TPSP:
    """
    Test cases for Model Training with 16 GPUs.
    Parallel Config:
        data parallel size = 8.
        tensor parallel size = 2.
        sequence parallel = True.
    """

    @staticmethod
    def setup_class():
        # update config tensor parallel size and sequence parallel
        command = f"sed -i 's/^.*tensor=.*/    tensor=2,/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)
        command = f"sed -i 's/^.*sequence_parallel=.*/    sequence_parallel=True,/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)

        # model training
        train()

        # print loss value
        print(f"cur_loss_list: {cur_loss_list}", flush=True)

    @staticmethod
    @pytest.mark.training_16GPU_8DP2TPSP
    def test_loss_spike_with_dp8_tp2_sp():
        check_loss_spike()

    @staticmethod
    @pytest.mark.training_16GPU_8DP2TPSP
    def test_loss_accuracy_with_dp8_tp2_sp():
        check_loss_accuracy()


class TestCaseTrain16GPUWith8DP2PP:
    """
    Test cases for Model Training with 16 GPUs.
    Parallel Config:
        data parallel size = 8.
        pipeline parallel size = 2.
    """

    @staticmethod
    def setup_class():
        # update config pipeline parallel size
        command = f"sed -i 's/^.*pipeline=.*/    pipeline=dict(size=2),/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)
        command = f"sed -i 's/^.*tensor=.*/    tensor=1,/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)

        # model training
        train()

        # print loss value
        print(f"cur_loss_list: {cur_loss_list}", flush=True)

    @staticmethod
    @pytest.mark.training_16GPU_8DP2PP
    def test_loss_spike_with_dp8_pp2():
        check_loss_spike()

    @staticmethod
    @pytest.mark.training_16GPU_8DP2PP
    def test_loss_accuracy_with_dp8_pp2():
        check_loss_accuracy()


class TestCaseTrain16GPUWith8DP2PPInterleaved:
    """
    Test cases for Model Training with 16 GPUs.
    Parallel Config:
        data parallel size = 8.
        pipeline parallel size = 2.
        interleaved scheduler = True.
    """

    @staticmethod
    def setup_class():
        # update config pipeline parallel size
        command = f"sed -i 's/^.*pipeline=.*/    pipeline=dict(size=2),/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)
        command = f"sed -i 's/^.*num_chunks=.*/    num_chunks=2,/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)
        command = f"sed -i 's/^.*tensor=.*/    tensor=1,/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=False)

        # model training
        train()

        # print loss value
        print(f"cur_loss_list: {cur_loss_list}", flush=True)

    @staticmethod
    @pytest.mark.training_16GPU_8DP2PP_Interleaved
    def test_loss_spike_with_dp8_pp2_interleaved():
        check_loss_spike()

    @staticmethod
    @pytest.mark.training_16GPU_8DP2PP_Interleaved
    def test_loss_accuracy_with_dp8_pp2_interleaved():
        check_loss_accuracy()


class TestCaseTrain16GPUWith8DP2PPInterleavedOverlap:
    """
    Test cases for Model Training with 16 GPUs.
    Parallel Config:
        data parallel size = 8.
        pipeline parallel size = 2.
        interleaved scheduler = True.
        interleaved overlap = True.
    """

    @staticmethod
    def setup_class():
        # update config pipeline parallel size
        command = f"sed -i 's/^.*pipeline=.*/    pipeline=dict(size=2, interleaved_overlap=True),/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)
        command = f"sed -i 's/^.*num_chunks=.*/    num_chunks=2,/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)
        command = f"sed -i 's/^.*tensor=.*/    tensor=1,/' {CONFIG_FILE_PATH}"
        subprocess.run(command, shell=True, check=True)

        # model training
        train()

        # print loss value
        print(f"cur_loss_list: {cur_loss_list}", flush=True)

    @staticmethod
    @pytest.mark.training_16GPU_8DP2PP_InterleavedOverlap
    def test_loss_spike_with_dp8_pp2_interleaved_overlap():
        check_loss_spike()

    @staticmethod
    @pytest.mark.training_16GPU_8DP2PP_InterleavedOverlap
    def test_loss_accuracy_with_dp8_pp2_interleaved_overlap():
        check_loss_accuracy()
