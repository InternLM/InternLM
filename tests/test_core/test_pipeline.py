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
from internlm.core.trainer import Trainer

from internlm.core.scheduler import (
    InterleavedPipelineScheduler,
    NonPipelineScheduler,
    PipelineScheduler,
    SchedulerHook,
)
from internlm.data.utils import unpack_data
from internlm.core.scheduler.pipeline_scheduler import get_tensor_shape
from internlm.core.context import global_context as gpc
from internlm.core.context import ParallelMode
from internlm.core.scheduler import SchedulerMetricHook
from internlm.model.metrics import AccPerplex 
from internlm.train import (
    get_train_data_loader,
    get_validation_data_loader,
    initialize_llm_profile,
    initialize_model,
    initialize_optimizer,
    load_new_batch,
    record_current_batch_training_metrics,
)
from internlm.core.engine import Engine
from internlm.model.loss import FlashGPTLMLoss
from internlm.core.gradient_handler import PipelineSharedModuleGradientHandler
from internlm.core.trainer import TrainState


class MlpModel(nn.Module):

    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 8)
        self.linear3 = nn.Linear(8, 8)
        self.linear4 = nn.Linear(8, 4)

    def forward(self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        print('MLP:', input_ids, input_ids.dtype, flush=True)
        input_ids = self.linear1(input_ids)
        input_ids = self.linear2(input_ids)
        input_ids = self.linear3(input_ids)
        input_ids = self.linear4(input_ids)
        return input_ids
    
config = Config(
    dict(
        parallel=dict(zero1=1, pipeline=dict(size=2, interleaved_overlap=False), sequence_parallel=False, tensor=1),
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
        beta2_scheduler = dict(
            init_beta2=0.95,
            c=0,
            cur_iter=-1,
        ),
        lr_scheduler = dict(
            total_steps=100,
            init_steps=0,  # optimizer_warmup_step
            warmup_ratio=0.01,
            eta_min=1e-5,
            last_epoch=-1,
        )
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



def exam_pipeline_parallel(args):
    rank, world_size = args
    dtype = torch.bfloat16
    
    build_environment(rank, world_size)
    seed_all(1024)
    
    torch_model = MlpModel().cuda()
    pp_model = copy.deepcopy(torch_model).to(dtype)
    
    
    tensor_shape = get_tensor_shape()
    
    scatter_gather = gpc.is_initialized(ParallelMode.TENSOR)
    
    if gpc.is_first_rank(ParallelMode.PIPELINE):
        print(rank, 'is first pp')
    
    

    scheduler_hooks = [
        SchedulerMetricHook(
            skip=False
        ),
    ]
    
    scheduler = PipelineScheduler(
        data_process_func=None,
        num_microbatches=gpc.config.data.micro_num,
        dtype=gpc.config.model["dtype"],
        tensor_shape=tensor_shape,
        scatter_gather_tensors=scatter_gather,
        scheduler_hooks=scheduler_hooks,
    )
    
    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=pp_model)
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=0)
    
    engine = Engine(
        model=pp_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        criterion=criterion,
        gradient_handlers=[],
        clip_grad_norm=gpc.config.hybrid_zero_optimizer.get("clip_grad_norm", 0.0),
    )
    
    scheduler.pre_processing(engine)
    engine.train()
    engine.zero_grad()
    
    input_list = [{'input_ids':torch.tensor([[0,1,2,3]]).cuda().to(dtype)},
                  torch.tensor([[1]]).cuda().to(torch.int64)]
    torch_input = torch.tensor([[0,1,2,3]]).cuda().to(torch.float32)
    torch_label = torch.tensor([[1]]).cuda().to(torch.int64)
    # print('label_shape:', input_list[1].shape)
    # input_list = [{'input_ids':torch.rand(1, 4).cuda()}, torch.rand(1, 4).cuda()]
    # input = input_list[0]
    # print(input)
    # output = torch_model(input)
    # print(output)
    print('local_rank:', gpc.get_local_rank(ParallelMode.PIPELINE), 'start schedule')
    _, _, loss = scheduler.forward_backward_step(engine, input_list, forward_only=False, return_loss=True, return_output_label=False)
    engine.step()
    print('local_rank:', gpc.get_local_rank(ParallelMode.PIPELINE), 'end schedule')
    torch_output = torch_model(input_ids=torch_input)
    torch_loss = criterion(torch_output, torch_label).unsqueeze(0)
    
    # if rank == 0:
    #     print('loss:', loss)
    #     print('torch_loss:', torch_loss)
    #loose_close(loss, torch_loss, dtype=dtype)
    torch_loss.backward()
    print('local_rank:', gpc.get_local_rank(ParallelMode.PIPELINE), 'everything3')
    
    
    


def test_pipeline_parallel():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(
            exam_pipeline_parallel,
            [[rank, 8] for rank in range(8)],
        )
        pool.close()
        pool.join()
        
        
if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_pipeline.py"])