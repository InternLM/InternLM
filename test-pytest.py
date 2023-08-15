import sys
import pytest
import torch
from internlm.model.embedding import Embedding1D
from flash_attn.modules.embedding import ParallelGPT2Embeddings
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.model.modeling_internlm import PackedFlashBaseLayer1D
import internlm
import multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import MixedFusedRMSNorm as RMSNorm
from internlm.model.linear import (
    FeedForward,
    RewardModelLinear,
    ScaleColumnParallelLinear,
)
from internlm.model.utils import gather_forward_split_backward
from internlm.model.modeling_internlm import PackedFlashInternLm1D



def initialize_distributed_env(config: str, launcher: str = "slurm", master_port: int = 8888, seed: int = 1024):
    """
    Initialize distributed environment for distributed training.

    Args:
        config (str): Config file path.
        launcher (str): Launcher for launching distributed environment, can be slurm or torch. "slurm" by default.
        master_port (str): The master port for distributed training. 8888 by default.
        seed (int, optional): Specified random seed for every process. 1024 by default.
    """

    torch.cuda.empty_cache()

    if launcher == "torch":
        internlm.launch_from_torch(config=config, seed=seed)
    else:
        assert launcher in ["slurm", "torch"], "launcher only support slurm or torch"
        

def build_environment(rank, world_size):
    import os
    os.environ['RANK'] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    initialize_distributed_env(config='./configs/7B_sft.py', launcher="torch", master_port=12345, seed=1024)
    

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


def check_block(args):
    rank, world_size = args
    build_environment(rank, world_size)
    
    device = torch.device("cuda")
    
    blocks = nn.ModuleList(
            [
                PackedFlashBaseLayer1D(
                    hidden_size=16, #768
                    num_attention_heads=2, #12
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
                for lid in range(1) #32
            ]
        )
    
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32).to(device)
    indexes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]).to(device)
    hidden_states = torch.tensor([[4, 118, 0, 1, 2, 3, 0, 1, 1, 97, 0, 0, 0, 0, 0, 0]]).to(device)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    
    embedding = Embedding1D(num_embeddings=400, embedding_dim=16, dtype=torch.bfloat16).to(device)
    
    s_w = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/weight_bfloat16_{rank}.pt').to(device)
    assert embedding.weight.shape == s_w.shape
    with torch.no_grad():
        embedding.weight = s_w
    
    
    hidden_states = embedding(hidden_states)
    hidden_states = hidden_states.squeeze(0).to(device).requires_grad_()
    
    s_h = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/hidden_{rank}.pt').to(device)
    rtol, atol = (1e-3, 5e-3) 
    assert torch.allclose(hidden_states, s_h, rtol=rtol, atol=atol, equal_nan=True)
    
    for _, block in enumerate(blocks):
        block = block.to(torch.bfloat16)
        block = block.to(device)
        result = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            indexes=indexes,
            inference_params=None,
            max_seqlen=max_seqlen,
        )
        
    #result = hidden_states
    standard = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/block_{rank}.pt').to(device)

    rtol, atol = (1e-3, 5e-3)                
    assert torch.allclose(result, standard, rtol=rtol, atol=atol, equal_nan=True)
    
    hidden_states.retain_grad()
    loss = torch.randn_like(result)
    
    result.backward(loss)
    grad = hidden_states.grad
    standard_grad = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/block_grad_{rank}.pt').to(device)
    
    rtol, atol = (1e-3, 5e-3) 
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol, equal_nan=True)


def check_norm(args):
    rank, world_size, norm_type, hidden_size, layer_norm_epsilon = args
    device = torch.device("cuda")
    build_environment(rank, world_size)
    
    if norm_type == "rmsnorm":
        norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
    else:
        norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
    norm = norm.to(device)
    hidden_states = torch.tensor([[ 8.3726,  1.9245,  5.5101,  1.0000],
                                [ 3.3474,  2.9582,  1.0000,  1.0000],
                                [ 8.3726,  1.2875,  5.5101,  1.0000],
                                [ 8.3726,  1.2875,  5.5101,  1.0000]], requires_grad=True).to(device)
    
    result = norm(hidden_states.float())
    standard = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/norm_{norm_type}.pt').to(device)
    
    rtol, atol = (1e-3, 5e-3)                
    assert torch.allclose(result, standard, rtol=rtol, atol=atol, equal_nan=True)
    
    hidden_states.retain_grad()
    loss = torch.randn_like(result)
    result.backward(loss)
    grad = hidden_states.grad
    
    standard_grad = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/norm_grad_{norm_type}.pt').to(device)
    rtol, atol = (1e-3, 5e-3) 
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol, equal_nan=True)


def check_head(args):
    rank, world_size, is_reward, hidden_size, vocab_size, embed_grad_scale = args
    device = torch.device("cuda")
    build_environment(rank, world_size)
    
    if is_reward:
        head_cls = RewardModelLinear
    else:
        head_cls = ScaleColumnParallelLinear
        
    head = head_cls(
        in_features=hidden_size,
        out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
        process_group=gpc.get_group(ParallelMode.TENSOR),
        bias=False,
        sequence_parallel=False,
        device=device,
        dtype=torch.bfloat16,
        weight_scale=embed_grad_scale,
    )
    
    head = head.to(torch.bfloat16)
    head = head.to(device)
    
    hidden_states = torch.tensor([[ 8.3726,  1.9245,  5.5101,  1.0000],
                                [ 3.3474,  2.9582,  1.0000,  1.0000],
                                [ 8.3726,  1.2875,  5.5101,  1.0000],
                                [ 8.3726,  1.2875,  5.5101,  1.0000]],
                                 dtype=torch.bfloat16, requires_grad=True).to(device)

    result = head(hidden_states)
    standard = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/head_{is_reward}.pt').to(device)

    rtol, atol = (1e-3, 5e-3)                
    assert torch.allclose(result, standard, rtol=rtol, atol=atol, equal_nan=True)
    
    hidden_states.retain_grad()
    loss = torch.randn_like(result)
    result.backward(loss)
    grad = hidden_states.grad
    
    standard_grad = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/head_grad_{is_reward}.pt').to(device)
    rtol, atol = (1e-3, 5e-3) 
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol, equal_nan=True)
    
    
def check_gather_forward(args):
    rank, world_size = args
    device = torch.device("cuda")
    build_environment(rank, world_size)
    
    hidden_states = torch.tensor([[ 8.3726,  1.9245,  5.5101,  1.0000],
                                [ 3.3474,  2.9582,  1.0000,  1.0000],
                                [ 8.3726,  1.2875,  5.5101,  1.0000],
                                [ 8.3726,  1.2875,  5.5101,  1.0000]], requires_grad=True).to(device)
    
    if rank == 0:
        print('gpc.config.parallel.tensor:', gpc.config.parallel.tensor)
    
    result = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=-1)
    standard = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/gather_forward_{gpc.config.parallel.tensor}.pt').to(device)
    
    rtol, atol = (1e-3, 5e-3)             
    assert torch.allclose(result, standard, rtol=rtol, atol=atol, equal_nan=True)
    
    loss = torch.randn_like(result)
    hidden_states.retain_grad()
    result.backward(loss)
    grad = hidden_states.grad
    if gpc.config.parallel.tensor > 1:
        standard_grad = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/gather_forward_grad_{gpc.config.parallel.tensor}_{rank}.pt').to(device)
    else:
        standard_grad = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/gather_forward_grad_{gpc.config.parallel.tensor}.pt').to(device)
    
        
    rtol, atol = (1e-3, 5e-3) 
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol, equal_nan=True)


def check_model(args):
    rank, world_size, embed_split_hidden = args
    device = torch.device("cuda")
    build_environment(rank, world_size)
    
    input_ids = torch.tensor([[4, 118, 0, 1, 2, 3, 0, 1, 1, 97, 0, 0, 0, 0, 0, 0]]).to(device)
    cu_seqlens = torch.tensor([[0, 8, 16]], dtype=torch.int32).to(device)
    indexes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]]).to(device)
    
    model = PackedFlashInternLm1D(num_layers=8,
                                  hidden_size=16,
                                  num_attention_heads=2,
                                  vocab_size=400,
                                  mlp_ratio=2.6666666666666665,
                                  attn_drop_rate=0,
                                  drop_rate=0,
                                  dtype=torch.bfloat16,
                                  checkpoint=False,
                                  checkpoint_fraction=1.0,
                                  layer_norm_epsilon=1e-05,
                                  first=True,
                                  last=True,
                                  embed_split_hidden=embed_split_hidden,
                                  embed_grad_scale=1,
                                  parallel_output=True,
                                  start_layer_idx=0,
                                  device=device,
                                  residual_in_fp32=False,
                                  norm_type='rmsnorm',
                                  is_reward=False,
                                  dropout_selective_checkpoint=True,
                                  use_scaled_init=True,
                                  use_swiglu=True)
    model.to(torch.bfloat16).to(device)
    
    states = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/model_states_{embed_split_hidden}_{rank}.pt')
    missing_k, unexpected_keys = model.load_state_dict(states, strict=False)
    if len(missing_k) != 0:
        print('missing_k', flush=True)
        print(missing_k, flush=True)
    if len(unexpected_keys) != 0:
        print('unexpected_keys', flush=True)
        print(unexpected_keys, flush=True)
        
    result = model(cu_seqlens=cu_seqlens, indexes=indexes, input_ids=input_ids)
    standard = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/model_result_{embed_split_hidden}_{rank}.pt')
    
    rtol, atol = (1e-3, 5e-3)                
    assert torch.allclose(result, standard, rtol=rtol, atol=atol, equal_nan=True)
    
    loss = torch.randn_like(result)
    result.backward(loss)
    
    if embed_split_hidden:
        grad = model.embedding.weight.grad
    else:
        grad = model.embedding.word_embeddings.weight.grad
    standard_grad = torch.load(f'/mnt/petrelfs/lijiaxing/train_llm/model_grad_{embed_split_hidden}_{rank}.pt')
    
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


@pytest.mark.block
def test_block():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_block, [[rank, 8] for rank in range(8)])
        pool.close()
        pool.join()


@pytest.mark.norm
@pytest.mark.parametrize('norm_type', ["rmsnorm"])
@pytest.mark.parametrize('hidden_size', [4])
@pytest.mark.parametrize('layer_norm_epsilon', [1e-05])       
def test_norm(norm_type, hidden_size, layer_norm_epsilon):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_norm, [[rank, 8, norm_type, hidden_size, layer_norm_epsilon] for rank in range(8)])
        pool.close()
        pool.join()


@pytest.mark.head
@pytest.mark.parametrize('is_reward', [True, False])
@pytest.mark.parametrize('hidden_size', [4])
@pytest.mark.parametrize('vocab_size', [4])
@pytest.mark.parametrize('embed_grad_scale', [1])       
def test_head(is_reward, hidden_size, vocab_size, embed_grad_scale):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_head, [[rank, 8, is_reward, hidden_size, vocab_size, embed_grad_scale] for rank in range(8)])
        pool.close()
        pool.join()
        

@pytest.mark.gather_forward
def test_gather_forward():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_gather_forward, [[rank, 8] for rank in range(8)])
        pool.close()
        pool.join()


@pytest.mark.model
@pytest.mark.parametrize('embed_split_hidden', [True, False])
def test_model(embed_split_hidden):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(check_model, [[rank, 8, embed_split_hidden] for rank in range(8)])
        pool.close()
        pool.join()
        

if __name__ == '__main__':
    pytest.main(['-s', '-q', 'test3.py']) #'-m embedding',