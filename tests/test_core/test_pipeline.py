import multiprocessing as mp

import pytest
import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config
from tests.test_core.utils import (
    MlpModel,
    MyLoss,
    build_environment,
    init_model_and_optim,
    loose_close,
    seed_all,
)

config = Config(
    dict(
        gradient_handler=[dict(type="PipelineSharedModuleGradientHandler")],
        parallel=dict(
            zero1=dict(size=1, fsdp=False),
            pipeline=dict(size=8, interleaved_overlap=False),
            sequence_parallel=False,
            tensor=1,
        ),
        model_type="INTERNLM",
        data=dict(seq_len=8, micro_num=16, micro_bsz=1, pack_sample_into_one=False, min_length=0, total_steps=9999),
        model=dict(
            dtype=torch.bfloat16,
            num_chunks=2,
            use_flash_attn=True,
        ),
        resume_tb_folder="",
        tensorboard_folder="",
        alert_address=None,
        monitor=dict(alert=dict(enable_feishu_alert=False, feishu_alert_address=None, light_monitor_address=None)),
        grad_scaler=dict(
            fp16=dict(
                initial_scale=1,
                min_scale=1,
                growth_interval=1,
            ),
            growth_factor=1.1,
            backoff_factor=0.9,
            max_scale=1,
            hysteresis=1,
        ),
        adam=dict(
            lr=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_beta2_c=0,
            adam_eps=1e-8,
            weight_decay=0.01,
        ),
        hybrid_zero_optimizer=dict(
            overlap_sync_grad=False,
            overlap_sync_param=False,
            reduce_bucket_size=512 * 1024 * 1024,
            clip_grad_norm=1.0,
        ),
        beta2_scheduler=dict(
            init_beta2=0.95,
            c=0,
            cur_iter=-1,
        ),
        lr_scheduler=dict(
            total_steps=100,
            init_steps=0,
            warmup_ratio=0.01,
            eta_min=1e-5,
            last_epoch=-1,
        ),
    )
)


def exam_pipeline_parallel(args):
    # init
    rank, world_size, micro_num, num_chunks, interleaved_overlap = args
    config.data.micro_num = micro_num
    config.model.num_chunks = num_chunks
    config.parallel.pipeline.interleaved_overlap = interleaved_overlap

    build_environment(rank, world_size, config)

    device = torch.device(f"cuda:{rank}")
    dtype = config.model["dtype"]
    seq_len = gpc.config.data.seq_len

    # set seed
    seed_all(1024)

    engine, scheduler = init_model_and_optim(32, num_chunks, dtype, micro_num, interleaved_overlap, tensor_shape=[1, 8])
    if scheduler is None:
        return

    scheduler.pre_processing(engine)
    engine.train()

    # create input
    x_list = []
    y_list = []
    for _ in range(micro_num):
        x_list.append(list(range(seq_len)))
        y_list.append(list(range(seq_len)))
    xs = torch.tensor(x_list).to(device).to(dtype)
    yx = torch.tensor(y_list).to(device).to(dtype)

    input_list = [{"input_ids": xs}, yx]

    # pp forward and backward
    output_list = []
    for _ in range(10):
        output, _, loss = scheduler.forward_backward_step(
            engine, input_list, forward_only=False, return_loss=True, return_output_label=True
        )
        output_list.append(output)

    engine.step()

    # torch related
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        torch_xs = torch.tensor(x_list).to(device).to(torch.float32)
        torch_ys = torch.tensor(y_list).to(device).to(torch.float32)
        torch_model = MlpModel(0, 32, "torch").to(device)
        torch_optimizer = torch.optim.AdamW(
            params=[{"params": torch_model.parameters(), "weight_decay": config.adam.weight_decay}],
            lr=config.adam.lr,
            betas=(config.adam.adam_beta1, config.adam.adam_beta2),
            eps=config.adam.adam_eps,
        )

        # check only forward logits
        first_output = output_list[0]
        for i in range(1, 10):
            assert torch.equal(first_output, output_list[i])

        # check output
        torch_output = torch_model(input_ids=torch_xs)  # pylint: disable=E1102
        loose_close(torch_output, output, dtype=dtype)

        torch_criterion = MyLoss().to(torch.float32)
        torch_loss = torch_criterion(torch_output, torch_ys) / micro_num  # pylint: disable=E1102
        torch_loss.backward()
        torch_optimizer.step()

        # check loss
        loose_close(torch_loss, loss[0], dtype=dtype)


@pytest.mark.parametrize("micro_num", [4, 8, 16])
@pytest.mark.parametrize("num_chunks", [1, 2, 4])
@pytest.mark.parametrize("interleaved_overlap", [True, False])
def test_pipeline_parallel(micro_num, num_chunks, interleaved_overlap):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(
            exam_pipeline_parallel,
            [[rank, 8, micro_num, num_chunks, interleaved_overlap] for rank in range(8)],
        )
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_pipeline.py"])
