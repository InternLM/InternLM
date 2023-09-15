import copy
import multiprocessing as mp
import random

import numpy as np
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
            dtype=torch.bfloat16,
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

    if dtype is torch.float32:
        rtol = 1.3e-6
        atol = 1e-5
    elif dtype is torch.bfloat16:
        rtol = 2e-2
        atol = 2e-2

    if isinstance(a, torch.Tensor):
        a = a.detach().to(dtype)
        b = b.detach().to(dtype)

    assert_close(a, b, rtol=rtol, atol=atol)


def init_optimizer_grouped_parameters(check_group, model):
    if check_group:
        optimizer_grouped_parameters = [
            {
                "params": list(model.parameters())[:2],
                "weight_decay": config.adam.weight_decay,
            },
            {
                "params": list(model.parameters())[2:],
                "weight_decay": config.adam.weight_decay,
            },
        ]
    else:
        optimizer_grouped_parameters = [{"params": model.parameters(), "weight_decay": config.adam.weight_decay}]

    return optimizer_grouped_parameters


def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def exam_hybrid_zero_optim_with_ddp(args):
    # init
    rank, world_size, zero_parallel, overlap_sync_param, overlap_sync_grad, micro_num, check_group, dtype = args
    # TODO: Need to test the combine of overlap param and group_params when ready
    # ParamBcastSyncHandler does not consider paramters in different optimizer group currently
    if overlap_sync_param and check_group:
        return
    config.parallel.zero1 = zero_parallel
    config.hybrid_zero_optimizer.overlap_sync_param = overlap_sync_param
    config.hybrid_zero_optimizer.overlap_sync_grad = overlap_sync_grad
    config.data.micro_num = micro_num
    config.model.dtype = dtype
    totel_step = 5
    if not overlap_sync_param:
        totel_step = 1

    build_environment(rank, world_size)
    seed_all(1024)

    # create models
    torch_model = MlpModel().cuda()
    zero_model = copy.deepcopy(torch_model).to(dtype)
    torch_model = DDP(torch_model.cuda(), static_graph=True).cuda()

    # create optimizer
    if config.hybrid_zero_optimizer.overlap_sync_param:
        param_bcast_sync_handler = ParamBcastSyncHandler(zero_model)
    else:
        param_bcast_sync_handler = None

    optimizer_grouped_parameters_zero = init_optimizer_grouped_parameters(check_group, zero_model)
    optimizer_grouped_parameters_torch = init_optimizer_grouped_parameters(check_group, torch_model)

    naive_optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters_zero,
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
        params=optimizer_grouped_parameters_torch,
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

            seed_all(1024 + rank)
            # create input
            input_data = torch.rand(16, 128).cuda()

            # zero-dp forward
            zero_output = zero_model(input_data.to(dtype))

            # torch-ddp forward
            torch_output = torch_model(input_data)

            # check output
            loose_close(zero_output, torch_output, dtype=dtype)

            # zero-dp backward
            zero_optimizer.backward(zero_output.mean())

            # torch-ddp backward
            if num == micro_num - 1:
                torch_output.mean().backward()
            else:
                with torch_model.no_sync():
                    torch_output.mean().backward()

        # zero-dp step
        zero_optimizer.step()

        # torch-ddp step
        torch_optimizer.step()

        # check grad
        if check_group:
            group1 = zip(list(torch_model.parameters())[:2], list(zero_model.parameters())[:2])
            group2 = zip(list(torch_model.parameters())[2:], list(zero_model.parameters())[2:])
            for torch_parm, zero_parm in group1:
                if zero_parm.grad is not None:
                    loose_close(torch_parm.grad, zero_parm.grad, dtype=dtype)
            for torch_parm, zero_parm in group2:
                if zero_parm.grad is not None:
                    loose_close(torch_parm.grad, zero_parm.grad, dtype=dtype)
        else:
            for torch_parm, zero_parm in zip(torch_model.parameters(), zero_model.parameters()):
                if zero_parm.grad is not None:
                    loose_close(torch_parm.grad, zero_parm.grad, dtype=dtype)

    torch.cuda.synchronize()
    # check updated param
    if check_group:
        group1 = zip(list(torch_model.parameters())[:2], list(zero_model.parameters())[:2])
        group2 = zip(list(torch_model.parameters())[2:], list(zero_model.parameters())[2:])
        for torch_parm, zero_parm in group1:
            loose_close(torch_parm, zero_parm, dtype=dtype)
        for torch_parm, zero_parm in group2:
            loose_close(torch_parm, zero_parm, dtype=dtype)
    else:
        for torch_parm, zero_parm in zip(torch_model.parameters(), zero_model.parameters()):
            loose_close(torch_parm, zero_parm, dtype=dtype)


def exam_hybrid_zero_optim_with_ckpt_load_save(args):
    # init
    rank, world_size, zero_parallel, check_group, dtype = args
    config.parallel.zero1 = zero_parallel
    config.parallel.dtype = dtype

    build_environment(rank, world_size)

    # create models
    zero_model = MlpModel().cuda().to(dtype)

    # create optimizer
    if config.hybrid_zero_optimizer.overlap_sync_param:
        param_bcast_sync_handler = ParamBcastSyncHandler(zero_model)
    else:
        param_bcast_sync_handler = None

    optimizer_grouped_parameters1 = init_optimizer_grouped_parameters(check_group, zero_model)
    optimizer_grouped_parameters2 = init_optimizer_grouped_parameters(check_group, zero_model)

    naive_optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters1,
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
        params=optimizer_grouped_parameters2,
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
        assert torch.equal(zero1_param, zero2_param)

    # check fp16 model weights
    for zero1_param, zero2_param in zip(
        zero_optimizer._fp16_param_groups.values(), zero_optimizer2._fp16_param_groups.values()
    ):
        assert zero1_param == zero2_param


zero_parallel_check_list = [-1, 1, 4]
overlap_sync_param_check_list = [True, False]
overlap_sync_grad_check_list = [True, False]
miro_num_check_list = [1, 2, 4]
check_group_list = [True, False]
dtype_list = [torch.float32, torch.bfloat16]


@pytest.mark.parametrize("zero_parallel", zero_parallel_check_list)
@pytest.mark.parametrize("overlap_sync_param", overlap_sync_param_check_list)
@pytest.mark.parametrize("overlap_sync_grad", overlap_sync_grad_check_list)
@pytest.mark.parametrize("micro_num", miro_num_check_list)
@pytest.mark.parametrize("check_group", check_group_list)
@pytest.mark.parametrize("dtype", dtype_list)
def test_hybrid_zero_optim_with_ddp(
    zero_parallel, overlap_sync_param, overlap_sync_grad, micro_num, check_group, dtype
):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(
            exam_hybrid_zero_optim_with_ddp,
            [
                [rank, 8, zero_parallel, overlap_sync_param, overlap_sync_grad, micro_num, check_group, dtype]
                for rank in range(8)
            ],
        )
        pool.close()
        pool.join()


@pytest.mark.parametrize("zero_parallel", zero_parallel_check_list)
@pytest.mark.parametrize("check_group", check_group_list)
@pytest.mark.parametrize("dtype", dtype_list)
def test_hybrid_zero_optim_with_ckpt_load_save(zero_parallel, check_group, dtype):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(
            exam_hybrid_zero_optim_with_ckpt_load_save,
            [[rank, 8, zero_parallel, check_group, dtype] for rank in range(8)],
        )
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_optimizer.py"])
