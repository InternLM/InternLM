import math

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

TOTAL_STEPS = 20
LOSS_DEVIATION_LIMIT = 0.5
BASELINE_LOSS_LIST = [
    11.64188003540039,
    7.92056131362915,
    6.936733245849609,
    6.104186058044434,
    5.954546928405762,
    5.975135803222656,
    5.03298807144165,
    4.648955821990967,
    4.212765693664551,
    3.7354514598846436,
    3.5600380897521973,
    3.525212526321411,
    3.2379345893859863,
    3.070716142654419,
    2.9927399158477783,
    2.8968098163604736,
    2.565525531768799,
    2.9832329750061035,
    2.8897438049316406,
    2.67700457572937,
]
cur_loss_list = []


def train():
    # init setting
    gpc.config.data.total_steps = TOTAL_STEPS
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
        assert loss is not None and not math.isnan(loss.item())
        global cur_loss_list
        cur_loss_list.append(loss.item())
        timer("fwd-bwd").stop()

        # update parameters, and returns (success_update, grad_norm)
        trainer_result = trainer.step()
        assert trainer_result is not None

        success_update, grad_norm_groups = trainer_result
        assert success_update, "Error: grad norm inf or nan occurs!"
        print(f"grad_norm_groups: {grad_norm_groups}")
        if success_update:  # update parameters successfully
            train_state.step_count += 1
        else:
            train_state.inf_nan_skip_batches += 1  # record the amount of updating parameters unsuccessfully.

        timer("one-batch").stop()


class TestCaseTraining:
    """
    Test cases for Model Training.
    """

    @staticmethod
    @pytest.mark.xdist_group("test_training")
    def test_loss():
        # initialize distributed environment
        initialize_distributed_env(config="./configs/7B_sft.py")
        assert hasattr(gpc, "config") and gpc.config is not None

        # model training
        train()

        # print loss value
        print(f"cur_loss_list: {cur_loss_list}", flush=True)

        for cur, target in zip(cur_loss_list, BASELINE_LOSS_LIST):
            assert (
                abs(cur - target) < LOSS_DEVIATION_LIMIT
            ), f"The loss value is abnormal, {target}->{cur}, please check it!"
