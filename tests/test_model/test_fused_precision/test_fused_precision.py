import multiprocessing as mp
from functools import partial

import pytest
import torch
from torch import nn

from internlm.core.naive_amp import NaiveAMPModel, set_fp32_attr_to_module
from internlm.model.modeling_internlm import PackedFlashBaseLayer1D
from internlm.train.utils import create_param_groups
from tests.test_model.test_model_internlm import build_environment, seed_all


def _pre_forward_hook_for_check(model, inputs):  # pylint: disable=W0613
    assert all(_.dtype == torch.float32 for _ in inputs)


def _post_forward_hook_for_check(model, inputs, outputs):  # pylint: disable=W0613
    if isinstance(outputs, tuple):
        assert all(_.dtype == torch.half for _ in outputs)
    else:
        assert outputs.dtype == torch.half


def check_fused_precision(args):
    # init
    rank, world_size = args
    device = torch.device("cuda")
    build_environment(rank, world_size)

    # fix seed
    seed_all(1024)
    # define model
    model = PackedFlashBaseLayer1D(
        hidden_size=16,  # 768
        num_attention_heads=2,  # 12
        mlp_ratio=2,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        dtype=torch.bfloat16,
        layer_norm_epsilon=1e-5,
        checkpoint=False,
        layer_idx=0,
        residual_in_fp32=False,
        device=device,
        norm_type="rmsnorm",
        dropout_selective_checkpoint=True,
        use_scaled_init=True,
        use_swiglu=True,
    )
    model = model.to(device)
    set_fp32_attr_to_module(model.norm1)
    model = NaiveAMPModel(
        model=model,
        output_to_fp32=True,
        dtype=torch.half,
        sync_buffer=False,
    )
    model.model.norm1.register_forward_pre_hook(partial(_pre_forward_hook_for_check))
    model.model.norm1.register_forward_hook(partial(_post_forward_hook_for_check))

    hidden_states = torch.rand(1, 1, 16).to(device).requires_grad_()

    # forward
    model(hidden_states)


class MlpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 1, bias=False).half()
        self.linear2 = nn.Linear(1, 4, bias=False).float()


def check_split_fused_group(args):
    # init
    rank, world_size = args
    device = torch.device("cuda")
    build_environment(rank, world_size)
    rtol, atol = (1e-3, 5e-3)

    # fix seed
    seed_all(1024)
    # define model
    model = MlpModel().to(device)
    groups = create_param_groups(model, weight_decay=0.05)

    standard_group = (
        {
            "name": "default",
            "params": [torch.Tensor([[0.3088, 0.2935, -0.2900, 0.4280]]).to(torch.float16).to(device).requires_grad_()],
            "weight_decay": 0.05,
        },
        {
            "name": "fp32",
            "params": [torch.Tensor([[0.6273], [0.4844], [-0.0463], [-0.0090]]).to(device).requires_grad_()],
            "weight_decay": 0.05,
        },
    )

    # check groups params
    for t1, t2 in zip(groups, standard_group):
        # assert t1["name"] == t2["name"]
        assert all(
            torch.allclose(p1, p2, rtol=rtol, atol=atol, equal_nan=True) for p1, p2 in zip(t1["params"], t2["params"])
        )


@pytest.mark.fused_precision
def test_fused_precision():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_fused_precision, [[rank, 8] for rank in range(8)])
        pool.close()
        pool.join()


@pytest.mark.split_groups
def test_split_fused_groups():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_split_fused_group, [[rank, 8] for rank in range(8)])
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_norm.py"])
