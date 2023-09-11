import pytest
import torch
from torch import nn
from tests.test_model.test_model_internlm import build_environment
from internlm.model.utils import try_import_RMSNorm
import multiprocessing as mp

RMSNorm = try_import_RMSNorm()

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

if __name__ == '__main__':
    pytest.main(['-s', '-q', 'test_norm.py'])