import os
import shutil
from subprocess import PIPE, STDOUT, Popen

import pytest
import torch

from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config
from internlm.solver.optimizer.hybrid_zero_optim import HybridZeroOptimizer
from internlm.utils.common import SingletonMeta

OSS_NAME = os.environ["OSS_BUCKET_NAME"]
OSS_IP = os.environ["OSS_IP"]
USER = os.environ["USER"]
JOB_NAME = "CI_TEST"
LOCAL_SAVE_PATH = "local:local_ckpt"

BOTO_SAVE_PATH = f"boto3:s3://{OSS_NAME}.{OSS_IP}/{USER}/{JOB_NAME}"
BOTO_SAVE_PATH_NO_PRFIX = f"s3://{OSS_NAME}.{OSS_IP}/{USER}/{JOB_NAME}/"

ASYNC_TMP_FOLDER = "./async_tmp_folder"


# 1B
init_config = Config(
    dict(
        parallel=dict(zero1=1, pipeline=dict(size=1, interleaved_overlap=False), sequence_parallel=False, tensor=1),
        model_type="INTERNLM",
        adam=dict(
            lr=1e-4,
        ),
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


def init_naive_model():
    # let MODEL_INITIALIZER to work
    import internlm.model.modeling_internlm  # noqa # pylint: disable=unused-import
    from internlm.core.naive_amp import NaiveAMPModel
    from internlm.utils.registry import MODEL_INITIALIZER

    model = MODEL_INITIALIZER.get_module(module_name=gpc.config.model_type)(**(init_config.model))
    model = NaiveAMPModel(
        model=model,
        output_to_fp32=False,
        dtype=torch.bfloat16,
        sync_buffer=False,
    )
    return model


def init_naive_optim(model):
    naive_optimizer = torch.optim.AdamW(
        params=[{"params": model.parameters(), "weight_decay": 0.01}],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return naive_optimizer


def init_hybrid_optim(model):
    naive_optimizer = torch.optim.AdamW(
        params=[{"params": model.parameters(), "weight_decay": 0.01}],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    optimizer = HybridZeroOptimizer(
        naive_optimizer,
        grad_scal_cfg=Config(
            dict(
                fp16=dict(
                    initial_scale=2**16,
                    min_scale=1,
                    growth_interval=1000,
                ),
                growth_factor=2,
                backoff_factor=0.5,
                max_scale=2**24,
                hysteresis=2,
            )
        ),
        zero_cfg=Config(
            dict(
                overlap_sync_grad=False,
                overlap_sync_param=False,
                reduce_bucket_size=512 * 1024 * 1024,
                clip_grad_norm=1.0,
            )
        ),
        param_bcast_sync_handler=None,
    )
    return optimizer


@pytest.fixture(autouse=True, scope="function")
def reset_singletons():
    SingletonMeta._instances = {}


def reset_seed():
    from internlm.core.context.random import _SEED_MANAGER

    _SEED_MANAGER.reset()


@pytest.fixture(scope="module")
def init_dist_and_model(rank=0, world_size=1):
    from internlm.initialize import initialize_distributed_env

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12377"
    initialize_distributed_env(config=init_config, launcher="torch", master_port=12377, args_check=False)

    # setup
    print("set up", flush=True)
    model = init_naive_model()
    # opim = init_naive_optim(model)
    opim = init_hybrid_optim(model)

    yield model, opim

    # teardown
    del model, opim
    print("teardown", flush=True)
    gpc.destroy()
    reset_seed()


def enter_flag(text):
    print(f"{text} begin!", flush=True)
    yield
    print(f"{text} end!", flush=True)


def del_tmp_file():
    try:
        shutil.rmtree(ASYNC_TMP_FOLDER, ignore_errors=True)
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree(LOCAL_SAVE_PATH.split(":")[1], ignore_errors=True)
    except FileNotFoundError:
        pass

    try:
        cmd = r"/mnt/petrelfs/share/sensesync --dryrun --deleteSrc cp " + BOTO_SAVE_PATH_NO_PRFIX + " / "
        with Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True) as output:
            results, presults = "", ""
            for line in iter(output.stdout.readline, b""):
                results += str(line.rstrip())
                presults += line.rstrip().decode() + "\n"
        print(presults, flush=True)
    except:  # noqa # pylint: disable=bare-except
        pass
