import copy
import multiprocessing as mp

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import internlm
from internlm.core.context.parallel_context import Config
from internlm.solver.optimizer import HybridZeroOptimizer
from internlm.solver.optimizer.utils import ParamBcastSyncHandler


class MlpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


config = Config(
    dict(
        parallel=dict(zero1=1, pipeline=dict(size=1, interleaved_overlap=False), sequence_parallel=False, tensor=1),
        model_type="INTERNLM",
        data=dict(seq_len=2048, micro_num=1, micro_bsz=1, pack_sample_into_one=False, min_length=0, total_steps=9999),
        model=dict(
            checkpoint=False,
            num_attention_heads=2,
            embed_split_hidden=True,
            vocab_size=103168,
            embed_grad_scale=1,
            parallel_output=True,
            hidden_size=1024,
            num_layers=2,
            mlp_ratio=1,
            apply_post_layer_norm=False,
            dtype=torch.bfloat16,
            norm_type="rmsnorm",
            layer_norm_epsilon=1e-5,
            use_flash_attn=True,
            num_chunks=1,
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
            overlap_sync_grad=True,
            overlap_sync_param=True,
            reduce_bucket_size=512 * 1024 * 1024,
            clip_grad_norm=1.0,
        ),
    )
)


def build_environment(rank, world_size):
    import os

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    torch.cuda.empty_cache()
    # launcher="torch"
    internlm.launch_from_torch(config=config, seed=1024)


def loose_close(a, b, dtype: torch.dtype = torch.float32):
    rtol, atol = (1e-3, 5e-3)
    if dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3

    assert_close(a, b, rtol=rtol, atol=atol)


def exam_hybrid_zero_optim_with_ddp(args):
    # init
    rank, world_size, zero_parallel, overlap_sync_param, overlap_sync_grad, micro_num = args
    config.parallel.zero1 = zero_parallel
    config.hybrid_zero_optimizer.overlap_sync_param = overlap_sync_param
    config.hybrid_zero_optimizer.overlap_sync_grad = overlap_sync_grad
    config.data.micro_num = micro_num
    totel_step = 5
    if overlap_sync_param == False:
        totel_step = 1

    build_environment(rank, world_size)

    # create models
    zero_model = MlpModel().cuda()
    torch_model = copy.deepcopy(zero_model)

    torch_model = DDP(torch_model.cuda(), static_graph=True).cuda()

    # create optimizer
    if config.hybrid_zero_optimizer.overlap_sync_param:
        param_bcast_sync_handler = ParamBcastSyncHandler(zero_model)
    else:
        param_bcast_sync_handler = None

    naive_optimizer = torch.optim.AdamW(
        params=[{"params": zero_model.parameters(), "weight_decay": config.adam.weight_decay}],
        lr=config.adam.lr,
        betas=(config.adam.adam_beta1, config.adam.adam_beta2),
        eps=config.adam.adam_eps,
    )

    zero_optimizer = HybridZeroOptimizer(
        naive_optimizer,
        grad_scal_cfg=config.grad_scaler,
        zero_cfg=config.hybrid_zero_optimizer,
        param_bcast_sync_handler=param_bcast_sync_handler,
    )

    torch_optimizer = torch.optim.AdamW(
        params=[{"params": torch_model.parameters(), "weight_decay": config.adam.weight_decay}],
        lr=config.adam.lr,
        betas=(config.adam.adam_beta1, config.adam.adam_beta2),
        eps=config.adam.adam_eps,
    )

    for _ in range(totel_step):
        zero_optimizer.zero_grad()
        torch_optimizer.zero_grad()
        zero_optimizer.skip_grad_reduce = True
        for num in range(micro_num):
            if num == micro_num - 1:
                zero_optimizer.skip_grad_reduce = False
            # create input
            input_data = torch.rand(16, 128).cuda()

            # zero-dp forward
            zero_output = zero_model(input_data)

            # torch-ddp forward
            torch_output = torch_model(input_data)

            # check output
            loose_close(zero_output, torch_output)

            # zero-dp backward
            zero_optimizer.backward(zero_output.mean())

            # torch-ddp backward
            if num == micro_num - 1:
                torch_output.mean().backward()
            else:
                with torch_model.no_sync():
                    torch_output.mean().backward()

        # check grad
        for torch_parm, zero_parm in zip(torch_model.parameters(), zero_model.parameters()):
            if torch_parm.grad is not None:
                loose_close(torch_parm.grad, zero_parm.grad)

        # zero-dp step
        zero_optimizer.step()

        # torch-ddp step
        torch_optimizer.step()

    torch.cuda.synchronize()
    # check updated param
    for torch_parm, zero_parm in zip(torch_model.parameters(), zero_model.parameters()):
        loose_close(torch_parm, zero_parm)


def exam_hybrid_zero_optim_with_ckpt_load_save(args):
    # init
    rank, world_size, zero_parallel = args
    config.parallel.zero1 = zero_parallel

    build_environment(rank, world_size)

    # create models
    zero_model = MlpModel().cuda()

    # create optimizer
    if config.hybrid_zero_optimizer.overlap_sync_param:
        param_bcast_sync_handler = ParamBcastSyncHandler(zero_model)
    else:
        param_bcast_sync_handler = None

    naive_optimizer = torch.optim.AdamW(
        params=[{"params": zero_model.parameters(), "weight_decay": config.adam.weight_decay}],
        lr=config.adam.lr,
        betas=(config.adam.adam_beta1, config.adam.adam_beta2),
        eps=config.adam.adam_eps,
    )

    zero_optimizer = HybridZeroOptimizer(
        naive_optimizer,
        grad_scal_cfg=config.grad_scaler,
        zero_cfg=config.hybrid_zero_optimizer,
        param_bcast_sync_handler=param_bcast_sync_handler,
    )

    naive_optimizer2 = torch.optim.AdamW(
        params=[{"params": zero_model.parameters(), "weight_decay": config.adam.weight_decay}],
        lr=config.adam.lr,
        betas=(config.adam.adam_beta1, config.adam.adam_beta2),
        eps=config.adam.adam_eps,
    )

    zero_optimizer2 = HybridZeroOptimizer(
        naive_optimizer2,
        grad_scal_cfg=config.grad_scaler,
        zero_cfg=config.hybrid_zero_optimizer,
        param_bcast_sync_handler=param_bcast_sync_handler,
    )

    # save and load states
    states = zero_optimizer.state_dict()
    zero_optimizer2.load_state_dict(states)

    # check fp32 model weights
    for zero1_param, zero2_param in zip(
        zero_optimizer._fp32_flat_param_groups_of_current_rank.values(),
        zero_optimizer2._fp32_flat_param_groups_of_current_rank.values(),
    ):
        loose_close(zero1_param, zero2_param)

    # check fp16 model weights
    for zero1_param, zero2_param in zip(
        zero_optimizer._fp16_param_groups.values(), zero_optimizer2._fp16_param_groups.values()
    ):
        loose_close(zero1_param, zero2_param)


zero_parallel_check_list = [-1, 1, 4]
overlap_sync_grad_check_list = [True, False]
overlap_sync_param_check_list = [True, False]
miro_num_check_list = [1, 2, 4]


@pytest.mark.parametrize("zero_parallel", zero_parallel_check_list)
@pytest.mark.parametrize("overlap_sync_param", overlap_sync_param_check_list)
@pytest.mark.parametrize("overlap_sync_grad", overlap_sync_grad_check_list)
@pytest.mark.parametrize("micro_num", miro_num_check_list)
def test_hybrid_zero_optim_with_ddp(zero_parallel, overlap_sync_param, overlap_sync_grad, micro_num):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(
            exam_hybrid_zero_optim_with_ddp,
            [[rank, 8, zero_parallel, overlap_sync_param, overlap_sync_grad, micro_num] for rank in range(8)],
        )
        pool.close()
        pool.join()


@pytest.mark.parametrize("zero_parallel", zero_parallel_check_list)
def test_hybrid_zero_optim_with_ckpt_load_save(zero_parallel):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(exam_hybrid_zero_optim_with_ckpt_load_save, [[rank, 8, zero_parallel] for rank in range(8)])
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_optimizer.py"])
