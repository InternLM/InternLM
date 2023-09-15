import multiprocessing as mp
import random

import numpy as np
import pytest
import torch
from torch import nn

import internlm
from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import Config
from internlm.core.context.parallel_context import global_context as gpc
from internlm.model.linear import RewardModelLinear, ScaleColumnParallelLinear
from internlm.model.modeling_internlm import PackedFlashBaseLayer1D
from internlm.model.utils import gather_forward_split_backward

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


def check_block(args):
    # init
    rank, world_size = args
    build_environment(rank, world_size)
    device = torch.device("cuda")
    rtol, atol = (1e-3, 5e-3)

    # fix seed
    seed_all(1024)

    # define block
    blocks = nn.ModuleList(
        [
            PackedFlashBaseLayer1D(
                hidden_size=4,  # 768
                num_attention_heads=2,  # 12
                mlp_ratio=2,
                attn_drop_rate=0.0,
                drop_rate=0.0,
                dtype=torch.bfloat16,
                layer_norm_epsilon=1e-5,
                checkpoint=lid < 0,
                layer_idx=lid + 0,  # This parameter is used for caching during generation
                residual_in_fp32=False,
                device=device,
                norm_type="rmsnorm",
                dropout_selective_checkpoint=True,
                use_scaled_init=True,
                use_swiglu=True,
            )
            for lid in range(4)  # 32
        ]
    )

    # create input
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32).to(device)  # [0, 8, 16]
    indexes = torch.tensor([0, 1, 0, 1]).to(device)  # [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
    hidden_states = torch.tensor([[0, 3, 2, 1]]).to(device)  # [[4, 118, 0, 1, 2, 3, 0, 1, 1, 97, 0, 0, 0, 0, 0, 0]]
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    hidden_states = torch.tensor(
        [
            [
                [-1.1620, 1.3113, 0.1507, 2.2698],
                [-1.2610, 1.0990, 0.3787, -0.3478],
                [1.4001, 1.1982, -0.6696, 0.3269],
                [1.3304, 1.2262, 1.0735, -1.1169],
            ]
        ]
    )

    hidden_states = hidden_states.squeeze(0).to(device).requires_grad_()

    # forward
    for _, block in enumerate(blocks):
        block = block.to(torch.bfloat16)
        block = block.to(device)
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            indexes=indexes,
            inference_params=None,
            max_seqlen=max_seqlen,
        )

    result = hidden_states
    standard_result = torch.tensor(
        [
            [-1.1621, 1.3111, 0.1509, 2.2697],
            [-1.2611, 1.0988, 0.3787, -0.3478],
            [1.4000, 1.1982, -0.6694, 0.3268],
            [1.3303, 1.2262, 1.0736, -1.1169],
        ]
    ).to(device)

    # check output
    assert torch.allclose(result, standard_result, rtol=rtol, atol=atol)

    hidden_states.retain_grad()
    loss = torch.randn_like(result)

    # backward
    result.backward(loss)

    grad = hidden_states.grad
    standard_grad = torch.tensor(
        [
            [0.7999, -0.2595, 0.2649, -1.3256],
            [0.7064, 0.0283, -0.5508, 0.6494],
            [-1.4657, -2.0316, 1.3776, 0.7211],
            [-0.6046, 0.4329, -0.1884, 1.1170],
        ]
    ).to(device)

    # check grad
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol)


def check_head(args):
    # init
    rank, world_size, is_reward = args
    device = torch.device("cuda")
    build_environment(rank, world_size)
    rtol, atol = (1e-3, 5e-3)
    hidden_size = 4
    vocab_size = 4
    embed_grad_scale = 1

    # fix seed
    seed_all(1024)

    # load standard
    if is_reward:
        head_cls = RewardModelLinear
        standard_result = torch.tensor([[3.5938], [1.0703], [3.6250], [3.6250]], dtype=torch.bfloat16).to(device)
        standard_grad = torch.tensor(
            [
                [-0.2246, 0.0164, -0.0591, 0.1660],
                [-0.5625, 0.0408, -0.1484, 0.4160],
                [-0.1758, 0.0128, -0.0464, 0.1299],
                [-0.4785, 0.0347, -0.1260, 0.3516],
            ],
            dtype=torch.bfloat16,
        ).to(device)
    else:
        head_cls = ScaleColumnParallelLinear
        standard_result = torch.tensor(
            [
                [3.5938, -2.2188, 2.0312, 3.5625],
                [1.0703, -1.1797, 1.1406, 1.6641],
                [3.6250, -2.0156, 1.7656, 3.4531],
                [3.6250, -2.0156, 1.7656, 3.4531],
            ],
            dtype=torch.bfloat16,
        ).to(device)
        standard_grad = torch.tensor(
            [
                [-0.2354, 0.0981, -0.2930, -0.6328],
                [0.2344, -0.2334, -0.0918, 0.1396],
                [-0.5898, -1.0156, -0.7070, 1.3750],
                [0.0242, -0.1494, 0.1206, -0.0427],
            ],
            dtype=torch.bfloat16,
        ).to(device)

    # define head
    head = head_cls(
        in_features=hidden_size,
        out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
        process_group=gpc.get_group(ParallelMode.TENSOR),
        bias=False,
        device=device,
        dtype=torch.bfloat16,
        weight_scale=embed_grad_scale,
    )

    head = head.to(torch.bfloat16)
    head = head.to(device)

    # create input
    hidden_states = torch.tensor(
        [
            [8.3726, 1.9245, 5.5101, 1.0000],
            [3.3474, 2.9582, 1.0000, 1.0000],
            [8.3726, 1.2875, 5.5101, 1.0000],
            [8.3726, 1.2875, 5.5101, 1.0000],
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    ).to(device)

    # forward
    result = head(hidden_states)

    # check output
    assert torch.allclose(result, standard_result, rtol=rtol, atol=atol)

    hidden_states.retain_grad()
    loss = torch.randn_like(result)

    # backward
    result.backward(loss)
    grad = hidden_states.grad

    # check grad
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol)


def check_gather_forward(args):
    # init
    rank, world_size, parallel_tensor = args
    assert parallel_tensor in [1, 2]
    config.parallel.tensor = parallel_tensor
    device = torch.device("cuda")
    build_environment(rank, world_size)
    rtol, atol = (1e-3, 5e-3)

    # fix seed
    seed_all(1024)

    # load standard
    if parallel_tensor == 1:
        standard_result = torch.tensor(
            [
                [8.3726, 1.9245, 5.5101, 1.0000],
                [3.3474, 2.9582, 1.0000, 1.0000],
                [8.3726, 1.2875, 5.5101, 1.0000],
                [8.3726, 1.2875, 5.5101, 1.0000],
            ]
        ).to(device)
        standard_grad = torch.tensor(
            [
                [-0.4461, 0.5602, -0.0625, -1.3609],
                [0.4353, 1.2988, 0.9595, -0.1144],
                [-0.7593, -0.4031, 0.2041, 1.4955],
                [0.5706, 0.9047, -0.6965, -0.3757],
            ]
        ).to(device)
    else:
        standard_result = torch.tensor(
            [
                [8.3726, 1.9245, 5.5101, 1.0000, 8.3726, 1.9245, 5.5101, 1.0000],
                [3.3474, 2.9582, 1.0000, 1.0000, 3.3474, 2.9582, 1.0000, 1.0000],
                [8.3726, 1.2875, 5.5101, 1.0000, 8.3726, 1.2875, 5.5101, 1.0000],
                [8.3726, 1.2875, 5.5101, 1.0000, 8.3726, 1.2875, 5.5101, 1.0000],
            ]
        ).to(device)
        if rank % 2 == 0:
            standard_grad = torch.tensor(
                [
                    [-0.4461, 0.5602, -0.0625, -1.3609],
                    [-0.7593, -0.4031, 0.2041, 1.4955],
                    [0.8093, 1.7580, 1.2996, -0.7545],
                    [1.0474, -0.5767, -1.0401, 0.8233],
                ]
            ).to(device)
        else:
            standard_grad = torch.tensor(
                [
                    [0.4353, 1.2988, 0.9595, -0.1144],
                    [0.5706, 0.9047, -0.6965, -0.3757],
                    [-1.3589, -0.7202, 0.6094, -0.8208],
                    [-1.0042, 0.3695, 0.2511, -0.2718],
                ]
            ).to(device)

    # create input
    hidden_states = torch.tensor(
        [
            [8.3726, 1.9245, 5.5101, 1.0000],
            [3.3474, 2.9582, 1.0000, 1.0000],
            [8.3726, 1.2875, 5.5101, 1.0000],
            [8.3726, 1.2875, 5.5101, 1.0000],
        ],
        requires_grad=True,
    ).to(device)

    # forward
    result = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=-1)

    # check output
    assert torch.allclose(result, standard_result, rtol=rtol, atol=atol)

    loss = torch.randn_like(result)
    hidden_states.retain_grad()

    # backward
    result.backward(loss)
    grad = hidden_states.grad

    # check grad
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol)


@pytest.mark.block
def test_block():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_block, [[rank, 8] for rank in range(8)])
        pool.close()
        pool.join()


@pytest.mark.head
@pytest.mark.parametrize("is_reward", [True, False])
def test_head(is_reward):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_head, [[rank, 8, is_reward] for rank in range(8)])
        pool.close()
        pool.join()


@pytest.mark.gather_forward
@pytest.mark.parametrize("parallel_tensor", [1, 2])
def test_gather_forward(parallel_tensor):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_gather_forward, [[rank, 8, parallel_tensor] for rank in range(8)])
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_model_internlm.py"])
