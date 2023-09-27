import multiprocessing as mp
import random

import numpy as np
import pytest
import torch

import internlm
from internlm.core.context.parallel_context import Config
from internlm.model.moe import MoE

config = Config(
    dict(
        parallel=dict(zero1=1, pipeline=dict(size=1, interleaved_overlap=False), sequence_parallel=False, tensor=2),
        model_type="INTERNLM_MoE",
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
            num_experts=4,
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
    os.environ["MASTER_PORT"] = "8889"
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


def check_moe(args):
    # init
    rank, world_size = args
    build_environment(rank, world_size)
    device = torch.cuda.current_device()
    rtol, atol = (1e-3, 5e-3)

    # fix seed
    seed_all(1024)

    # define moe block
    moe_block = MoE(
        hidden_size=8,
        num_experts=4,
        ep_size=4,
        k=2,
        device=device,
        dtype=torch.bfloat16,
    ).to(device=device, dtype=torch.bfloat16)

    # create input
    hidden_states = torch.tensor(
        [
            [ 0.0080,  1.4003, -0.0911,  1.5041, -0.9852,  0.0073,  0.8122,  0.5846],
            [-0.9325,  1.1439,  0.1247,  0.9126, -1.8346, -1.4484, -1.0012, -0.2540],
            [ 1.0282, -0.7587, -1.4941,  0.0623, -2.6417, -0.6424,  0.1384, -0.8128],
            [ 0.3021, -0.4711,  0.0220,  1.0690, -1.2214, -1.1801,  1.1554, -0.6620],
        ],
        dtype=torch.bfloat16,
    )
    hidden_states = hidden_states.squeeze(0).to(device).requires_grad_()

    # forward
    result, moe_loss, _ = moe_block(hidden_states)

    # check forward
    standard_result = torch.tensor(
        [
            [-0.1143, -0.0064, -0.0269,  0.0374, -0.0854,  0.0294,  0.1895,  0.0200],
            [-0.1650, -0.6523,  0.4199,  0.3574,  0.2227, -0.0957, -0.4121,  0.6055],
            [-0.0214, -0.5508,  0.5977, -0.6406, -0.6562, -0.6172,  0.5117,  0.0664],
            [ 0.3223, -0.0649,  0.2891, -0.1855, -0.2334,  0.5078,  0.3789,  0.4941],
        ],
        dtype=torch.bfloat16,
    ).to(device)

    # check output
    assert torch.allclose(result, standard_result, rtol=rtol, atol=atol)

    # backward
    hidden_states.retain_grad()
    loss = moe_loss + torch.randn_like(result)

    result.backward(loss)
    grad = hidden_states.grad

    # check backward
    standard_grad = torch.tensor(
        [
            [ 0.5898, -0.6758,  0.0021,  0.5469, -0.6172, -0.1289,  0.5234, -0.5391],
            [ 0.2539, -0.6016,  0.0271,  0.7109,  0.1162, -0.5781, -0.4258, -0.5781],
            [-1.7734,  0.5312,  1.7031,  0.3672,  1.0781, -1.2891,  0.5625,  1.1406],
            [ 0.1328, -0.6250,  0.3945,  0.8633, -0.4805, -0.4023,  0.5039, -0.1914],
        ],
        dtype=torch.bfloat16,
    ).to(device)

    # check grad
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol)


@pytest.mark.moe
def test_moe():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_moe, [[rank, 8] for rank in range(8)])
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_moe.py"])
