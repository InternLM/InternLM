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
from internlm.solver.pipeline_utils import partition_uniform


import torch.distributed as dist

class MlpModel(nn.Module):

    def __init__(self, start, end, type=None):
        super().__init__()
        self.part = [start , end]
        self.blocks = nn.ModuleList([nn.Linear(8, 8, bias=False) for lid in range(end -start)])
        self.type = type
        if gpc.is_first_rank(ParallelMode.PIPELINE):
            print(f'{gpc.get_global_rank()}: self.part={self.part}', flush=True)

    def forward(self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        # print(gpc.get_global_rank(), 'hidden_states:', hidden_states, flush=True)
        if self.type != 'torch' and not gpc.is_first_rank(ParallelMode.PIPELINE):
            input_ids = hidden_states
            
        # print(f'pp stage: {gpc.get_local_rank(ParallelMode.PIPELINE)} MLP {self.part} fwd:', input_ids.shape, flush=True)
        # print(gpc.get_global_rank(), 'len_blocsk:', len(self.blocks), flush=True)
        # current_device = torch.cuda.current_device()
        # print(gpc.get_global_rank(), 'current_device:', current_device, flush=True)
        # input_ids = input_ids.to(current_device)
        # print(gpc.get_global_rank(), 'mlp_input_data:', input_ids, input_ids.shape, type(input_ids), flush=True)
        for i in range(self.part[1] - self.part[0]):
            input_ids = self.blocks[i](input_ids)
        return input_ids
        # x = self.blocks[0](input_ids)
        # x = self.blocks[0](x)
        # print(gpc.get_global_rank(), 'mlp_output_data:', x, x.shape, flush=True)
        # return x

config = Config(
    dict(
        HIDDEN_SIZE=8,
        SEQ_LEN=8,
        gradient_handler=[dict(type="PipelineSharedModuleGradientHandler")],
        parallel=dict(zero1=1, pipeline=dict(size=8, interleaved_overlap=True), sequence_parallel=False, tensor=1),
        model_type="INTERNLM",
        data=dict(seq_len=8, micro_num=16, micro_bsz=1, pack_sample_into_one=False, min_length=0, total_steps=9999),
        model=dict(
            dtype=torch.bfloat16,
            num_chunks=2,
            hidden_size=8,
            use_flash_attn=True,
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
    os.environ["MASTER_PORT"] = "33333"
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



def _build_generic_model_1d(num_layers, num_chunks, device=torch.device("cuda"), **kwargs):
    """
    build generic model 1d

    Args:
        num_layers (int): The number of layer.
        num_chunks (int): The number of partitions in pipeline parallel.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]
    if gpc.is_rank_for_log():
        print(f"The layer sharding is {all_parts}.", flush=True)

    models = []
    for start, end in parts:
        models.append(MlpModel(start, end).cuda())
    torch.distributed.barrier()
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        loss = torch.nn.MSELoss(reduction='sum')
        print(logits, flush=True)
        print(labels, flush=True)
        return loss(logits, labels)
    
def exam_pipeline_parallel(args):
    import os
    # rank, world_size = args

    rank = os.environ["RANK"] 
    world_size = os.environ["WORLD_SIZE"] 

    build_environment(rank, world_size)
    local_rank = int(os.environ["LOCAL_RANK"])
    print('rank_com:', rank, local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # print('device_id:', device)
    # torch.cuda.set_device(device)
    seed_all(1024)
    dtype=gpc.config.model["dtype"]

    
    # pp_model = copy.deepcopy(torch_model).to(dtype)
    pp_model = _build_generic_model_1d(num_layers=16, num_chunks=gpc.config.model.num_chunks)
    pp_model = pp_model.to(dtype)
    print(gpc.get_global_rank(), 'pp_model', pp_model)
    

    scheduler_hooks = [
        SchedulerMetricHook(
            skip=True
        ),
    ]

    micro_num = gpc.config.data.micro_num
    seq_len = gpc.config.data.seq_len
    gpc.config.NUM_MICRO_BATCHES = micro_num
    
    communication_overlap = gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
    print(f'communication_overlap={communication_overlap}')
    scheduler = InterleavedPipelineScheduler(
        num_microbatches=micro_num,
        num_chunks=gpc.config.model.num_chunks,
        dtype=gpc.config.model["dtype"],
        tensor_shape=get_tensor_shape(),
        scatter_gather_tensors=False,
        scheduler_hooks=scheduler_hooks,
        communication_overlap=communication_overlap,
    )
    # scheduler = PipelineScheduler(
    #     data_process_func=None,
    #     num_microbatches=micro_num,
    #     dtype=dtype,
    #     tensor_shape=None,
    #     scatter_gather_tensors=False,
    #     scheduler_hooks=scheduler_hooks,
    # )

    print(f"gpc.config.hybrid_zero_optimizer: {gpc.config.hybrid_zero_optimizer}", flush=True)
    # optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=pp_model)
    # criterion = FlashGPTLMLoss(parallel_output=False, label_smoothing=0)

    # from internlm.solver.optimizer.hybrid_zero_optim import BaseOptimizer
    # optimizer = BaseOptimizer(torch.optim.AdamW(
    #     params=[{"params": pp_model.parameters()}],
    #     lr=1e-4,
    #     betas=(0.9, 0.95),
    #     eps=1e-8,
    # ))
    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=pp_model)

    engine = Engine(
        model=pp_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        criterion=MyLoss().to(dtype),
        gradient_handlers= [PipelineSharedModuleGradientHandler(model=pp_model, optimizer=optimizer)],
        clip_grad_norm=gpc.config.hybrid_zero_optimizer.get("clip_grad_norm", 0.0),
    )

    scheduler.pre_processing(engine)
    engine.train()
    # engine.zero_grad()

    x_list = []
    y_list = []
    for _ in range(micro_num):
        x_list.append([i for i in range(seq_len)])
        y_list.append([i for i in range(seq_len)])
    torch_xs = torch.tensor(x_list).to(device).to(torch.float32)
    torch_ys = torch.tensor(y_list).to(device).to(torch.float32)
    xs = torch.tensor(x_list).to(device).to(dtype)
    yx = torch.tensor(y_list).to(device).to(dtype)
    # xs.requires_grad_()
    # yx.requires_grad_()
    print(xs.shape, yx.shape, flush=True)
    input_list = [{'input_ids':xs}, yx]

    # torch_input = torch.tensor([[0,1,2,3]]).to(device).to(torch.float32)
    # torch_label = torch.tensor([[1]]).to(device).to(torch.int64)
    # print('label_shape:', input_list[1].shape)
    # input_list = [{'input_ids':torch.rand(1, 4).cuda()}, torch.rand(1, 4).cuda()]
    # input = input_list[0]
    # print(input)
    # output = torch_model(input)
    # print(output)
    print('local_rank:', gpc.get_local_rank(ParallelMode.PIPELINE), 'start schedule', flush=True)
    output, label, loss = scheduler.forward_backward_step(engine, input_list, forward_only=False, return_loss=True, return_output_label=True)
    print('local_rank:', gpc.get_local_rank(ParallelMode.PIPELINE), 'end schedule', flush=True)

    #dist.barrier()
    torch.cuda.synchronize()
    engine.step()
    torch.cuda.synchronize()
    
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        print('torch begin')
        torch_model = MlpModel(0, 16, 'torch').to(device)
        # torch_model = DDP(torch_model, static_graph=True)
        print(gpc.get_global_rank(), 'torch_model', torch_model)
        torch_optimizer = torch.optim.AdamW(
            params=[{"params": torch_model.parameters(), "weight_decay": config.adam.weight_decay}],
            lr=config.adam.lr,
            betas=(config.adam.adam_beta1, config.adam.adam_beta2),
            eps=config.adam.adam_eps,
        )
        torch_output = torch_model(input_ids=torch_xs)
        criterion = MyLoss().to(torch.float32)
        torch_loss = criterion(torch_output, torch_ys) / micro_num
        torch_loss.backward()
        torch_optimizer.step()
        print(gpc.get_global_rank(), 'test_torch:', 'torch_output:', torch_output, 'torch_loss:', torch_loss)
        print(gpc.get_global_rank(), 'test_pp:', 'output:', output, 'label:', label, 'loss:', loss)
        loose_close(torch_output, output, dtype=dtype)
        loose_close(torch_loss, loss[0], dtype=dtype)
        print(gpc.get_global_rank(), 'assert_ok')

    # if rank == 0:
    #     print('loss:', loss)
    #     print('torch_loss:', torch_loss)
    #loose_close(loss, torch_loss, dtype=dtype)
    # torch_loss.backward()
    print('local_rank:', gpc.get_local_rank(ParallelMode.PIPELINE), 'everything3')





# def test_pipeline_parallel():
#     ctx = mp.get_context("spawn")
#     with ctx.Pool(processes=8) as pool:
#         pool.map(
#             exam_pipeline_parallel,
#             [[rank, 8] for rank in range(8)],
#         )
#         pool.close()

#         pool.join()


if __name__ == "__main__":
    # pytest.main(["-s", "-q", "test_pipeline.py"])
    exam_pipeline_parallel(None)
