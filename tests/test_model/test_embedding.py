import torch
import pytest
from tests.test_model.test_model_internlm import build_environment
from internlm.model.embedding import Embedding1D
from flash_attn.modules.embedding import ParallelGPT2Embeddings
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
import multiprocessing as mp

def check_embedding(args):
    rank, world_size, embed_split_hidden, vocab_size, hidden_size = args
    device=torch.device("cuda")
    build_environment(rank, world_size)
    
    if embed_split_hidden:
        embedding = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
    else:
        embedding = ParallelGPT2Embeddings(
            embed_dim=hidden_size,
            vocab_size=vocab_size,
            max_position_embeddings=-1,
            process_group=gpc.get_group(ParallelMode.TENSOR),
            padding_idx=None,
            sequence_parallel=False,
            device=device,
            dtype=torch.float,
        )
    embedding = embedding.to(device)
    if embed_split_hidden:
        s_w = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/weight_{rank}.pt').to(device)
        assert embedding.weight.shape == s_w.shape
        with torch.no_grad():
            embedding.weight = s_w
    
    
    standard = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/embedding_{embed_split_hidden}_{rank}.pt').to(device)
    
    input_ids = torch.tensor([[4, 118, 0, 1, 2, 3, 0, 1, 1, 97, 0, 0, 0, 0, 0, 0]]).to(device)
    result = embedding(input_ids)
 
    rtol, atol = (1e-3, 5e-3)                
    assert torch.allclose(result, standard, rtol=rtol, atol=atol, equal_nan=True)
    
    loss = torch.randn_like(result)
    result.backward(loss)
    
    if embed_split_hidden:
        grad = embedding.weight.grad
    else:
        grad = embedding.word_embeddings.weight.grad
    
    standard_grad = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/embedding_grad_{embed_split_hidden}_{rank}.pt').to(device)
    rtol, atol = (1e-3, 5e-3) 
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol, equal_nan=True)

    
@pytest.mark.embedding
@pytest.mark.parametrize('embed_split_hidden', [True, False])
@pytest.mark.parametrize('vocab_size', [400]) #, 50304, 103168
@pytest.mark.parametrize('hidden_size', [16]) #, 768, 4096
# @pytest.mark.parametrize('vocab_size, hidden_size', [(100, 16)])
def test_embedding(embed_split_hidden, vocab_size, hidden_size):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_embedding, [[rank, 8, embed_split_hidden, vocab_size, hidden_size] for rank in range(8)])
        pool.close()
        pool.join()

if __name__ == '__main__':
    pytest.main(['-s', '-q', 'test_embedding.py'])