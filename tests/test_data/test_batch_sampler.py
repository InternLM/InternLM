import multiprocessing as mp

import numpy as np
import pytest
import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

# from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import Config
from internlm.core.trainer import TrainState
from internlm.train import (
    get_train_data_loader,
    get_validation_data_loader,
    load_new_batch,
)
from internlm.utils.evaluation import (
    switch_evaluation_no_pipeline_scheduler,
    switch_evaluation_pipeline_scheduler,
)

# from internlm.core.context.parallel_context import global_context as gpc
from tests.test_core.utils import build_environment, init_model_and_optim

micro_bszs = [1, 2]
use_flash_attens = [True, False]
answers = [[1] * 8, [1, 1, 1, 1, 2, 2, 2, 2], [4] * 8, [2, 2, 4, 4, 6, 6, 8, 8]]
test_case_group = [
    # format: micro_nums, rampup_batch_size, should sccuess, answer, pp size, sql len
    (1, "1 1 1", True, answers[0], 1, 8),
    (4, "1 1 4", True, answers[1], 1, 8),
    (4, None, True, answers[2], 1, 8),
    (8, "2 2 2", True, answers[3], 1, 8),
    (8, "2 2 2", True, answers[3], 2, 8),
]


class DummyTrainer:
    def __init__(self, scheduler) -> None:
        self.schedule = scheduler


def do_warmup(args):
    rank, worldsize, init_config, should_sccuess, answer = args
    build_environment(rank, worldsize, init_config)
    gpc.config.model.num_chunks = 1 if gpc.get_world_size(ParallelMode.PIPELINE) == 1 else 2
    engine, scheduler = init_model_and_optim(
        8,
        gpc.config.model.num_chunks,
        torch.bfloat16,
        init_config.data.micro_num,
        True,
        tensor_shape=[1, 8],  # can't use get_tensor_shape becase we use toy model.
        init_optim=False,
        embedding=True,
    )
    scheduler.pre_processing(engine)
    engine.train()
    trainer = DummyTrainer(scheduler)

    try:
        train_dl, _ = get_train_data_loader(num_worker=0)
        val_dls = get_validation_data_loader(num_worker=0)
    except Exception as e:
        assert should_sccuess is False, f"{e}"
    else:
        assert should_sccuess is True

    # initialize and resume train state
    train_state = TrainState(gpc.config, train_dl.batch_sampler)
    # transfer the train data loader into train data iterator
    train_iter = iter(train_dl)

    micro_bsz = gpc.config.data.micro_bsz
    sql = gpc.config.data.seq_len

    consumed_token = 0  # Token consumed
    packed_length = micro_bsz * sql
    for i in range(init_config.data.total_steps):
        batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter, train_state=train_state)
        type_ids_shape = batch[0]["type_ids"].shape
        input_ids_shape = batch[0]["input_ids"].shape

        tokens_num = np.prod(input_ids_shape)

        # If not use fa, 'type_ids' is unpcaked when load_new_batch is calling.
        # However, 'input_ids' is unpcaked in pp/nopp engine.
        if not init_config.model.use_flash_attn:
            assert type_ids_shape == torch.Size(
                [answer[i], micro_bsz, sql]
            ), f"iter:{i}, type_ids_shape: {type_ids_shape} != {torch.Size([answer[i], micro_bsz, sql])}"
        else:
            assert type_ids_shape == torch.Size(
                [answer[i], packed_length]
            ), f"iter:{i}, type_ids_shape: {type_ids_shape} != {torch.Size([answer[i], packed_length])}"

        assert input_ids_shape == torch.Size(
            [answer[i], packed_length]
        ), f"iter:{i}, input_ids_shape: {input_ids_shape} != {torch.Size([answer[i], packed_length])}"

        if gpc.get_global_rank() == 0:
            print(
                f"iter:{i}",
                f"pp size: {gpc.get_world_size(ParallelMode.PIPELINE)}",
                f"use_flash_attn:{gpc.config.model.use_flash_attn}",
                f"micro_bsz:{micro_bsz}",
                f"input shape: {batch[0]['type_ids'].shape}",
                f"rampup_batch_size: {gpc.config.data.rampup_batch_size}",
                f"tokens_num: {tokens_num}",
                flush=True,
            )

        consumed_token += tokens_num
        batch[0].pop("type_ids", None)
        batch[0]["input_ids"] = batch[0]["input_ids"].to(torch.bfloat16)

        scheduler.forward_backward_step(engine, batch, forward_only=True, return_loss=False, return_output_label=False)
        assert (
            tokens_num == answer[i] * gpc.config.data.seq_len * micro_bsz
        ), f"{tokens_num} == {answer[i] * gpc.config.data.seq_len * micro_bsz}"

    # test no-packed datasets.
    for _, val_dl in val_dls.items():
        for _, batch in enumerate(val_dl):
            if gpc.is_using_pp():
                total_val_bsz = len(batch[1])
                batch[0]["input_ids"] = batch[0]["input_ids"].to(torch.bfloat16)
                assert total_val_bsz % micro_bsz == 0
                num_microbatches = total_val_bsz // micro_bsz
                tensor_shape = torch.Size([micro_bsz, batch[0]["input_ids"].shape[1]])  # toy model hidden size is 8.
                with switch_evaluation_pipeline_scheduler(
                    trainer=trainer,
                    num_microbatches=num_microbatches,
                    tensor_shape=tensor_shape,
                    metric_hook_list=[],
                ):
                    scheduler.forward_backward_step(
                        engine, batch, forward_only=True, return_loss=False, return_output_label=False
                    )
            else:
                total_val_bsz = len(batch[1])
                batch[0]["input_ids"] = batch[0]["input_ids"].to(torch.bfloat16)
                assert total_val_bsz % micro_bsz == 0
                grad_accum_size = total_val_bsz // micro_bsz
                with switch_evaluation_no_pipeline_scheduler(
                    trainer=trainer,
                    grad_accum_size=grad_accum_size,
                    metric_hook_list=[],
                ):
                    scheduler.forward_backward_step(
                        engine, batch, forward_only=True, return_loss=False, return_output_label=False
                    )


@pytest.mark.parametrize("use_flash_atten_case", use_flash_attens)
@pytest.mark.parametrize("group_case", test_case_group)
@pytest.mark.parametrize("micro_bsz_case", micro_bszs)
def test_warmup(use_flash_atten_case, group_case, micro_bsz_case):
    ctx = mp.get_context("spawn")
    # print(pp_size_case, use_flash_atten_case, group_case, micro_bsz_case, flush=True)

    config = Config(
        dict(
            parallel=dict(
                zero1=dict(size=1, fsdp=False),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence_parallel=False,
                tensor=1,
            ),
            data=dict(
                train_folder=None,
                valid_folder=None,
                valid_micro_num=4,
                pack_sample_into_one=False,
                min_length=0,
                total_steps=8,
            ),
            model=dict(
                dtype=torch.bfloat16,
            ),
            adam=dict(lr=1e-4),
            resume_tb_folder=None,
            tensorboard_folder=None,
        )
    )

    config.data.seq_len = group_case[5]
    config.parallel.pipeline.size = group_case[4]
    config.model.use_flash_attn = use_flash_atten_case
    config.data.micro_bsz = micro_bsz_case
    config.data.micro_num = group_case[0]
    config.data.gradient_accumulation = config.data.micro_num
    config.data.rampup_batch_size = group_case[1]
    config.data.packed_length = micro_bsz_case * config.data.seq_len
    should_sccuess = group_case[2]
    answer = group_case[3]

    with ctx.Pool(processes=8) as pool:
        pool.map(do_warmup, [[rank, 8, config, should_sccuess, answer] for rank in range(8)])
        pool.close()
        pool.join()
