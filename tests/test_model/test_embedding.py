import multiprocessing as mp

import pytest
import torch

from internlm.model.embedding import Embedding1D
from tests.test_model.test_model_internlm import build_environment, seed_all


def check_embedding(args):
    # init
    rank, world_size = args
    device = torch.device("cuda")
    build_environment(rank, world_size)
    rtol, atol = (1e-3, 5e-3)
    vocab_size = 4
    hidden_size = 2

    # fix seed
    seed_all(1024)

    # define embedding
    embedding = Embedding1D(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        padding_idx=None,
    )

    embedding.weight.data.copy_(torch.randn(vocab_size, hidden_size))
    embedding = embedding.to(device)

    # create input
    input_ids = torch.tensor([[0, 2], [1, 3]]).to(device)
    result = embedding(input_ids)

    standard_list = [[[-1.4837, 0.2671], [0.6002, -0.5496]], [[-1.8337, -0.1047], [1.0391, 0.2261]]]
    standard_result = torch.tensor(standard_list).to(device)

    # check output
    assert torch.allclose(result, standard_result, rtol=rtol, atol=atol, equal_nan=True)

    loss = torch.randn_like(result)

    # backward
    result.backward(loss)

    grad = embedding.weight.grad
    standard_glist = [[-0.4461, 0.5602], [0.4353, 1.2988], [-0.0625, -1.3609], [0.9595, -0.1144]]
    standard_grad = torch.tensor(standard_glist).to(device)

    # check grad
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.embedding
def test_embedding():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_embedding, [[rank, 8] for rank in range(8)])
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_embedding.py"])
