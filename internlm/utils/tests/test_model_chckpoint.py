import os
import shutil
from subprocess import PIPE, STDOUT, Popen

import pytest
import torch

from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config
from internlm.core.trainer import TrainState
from internlm.solver.optimizer.hybrid_zero_optim import HybridZeroOptimizer
from internlm.utils.model_checkpoint import CheckpointManager
from internlm.utils.storage_manager import (
    init_storage_manager,
    wait_async_upload_finish,
)
from internlm.utils.tests.common_fixture import (  # noqa # pylint: disable=unused-import
    init_dist_and_model,
    reset_singletons,
)

TOTAL_STEP = 6

CKPT_EVERY = 4
SNPASHOT_EVERY = 2
OSS_NAME = os.environ["OSS_BUCKET_NAME"]
OSS_IP = os.environ["OSS_IP"]
USER = os.environ["USER"]
JOB_NAME = "CI_TEST"
LOCAL_SAVE_PATH = "local:local_ckpt"

BOTO_SAVE_PATH = f"boto3:s3://{OSS_NAME}.{OSS_IP}/{USER}/{JOB_NAME}"
BOTO_SAVE_PATH_NO_PRFIX = f"s3://{OSS_NAME}.{OSS_IP}/{USER}/{JOB_NAME}/"

ASYNC_TMP_FOLDER = "./async_tmp_folder"


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
    except FileNotFoundError:
        pass


ckpt_config_list = [
    # Old interface format
    dict(
        enable_save_ckpt=True,
        save_ckpt_folder=BOTO_SAVE_PATH,
        load_optimizer=True,
        checkpoint_every=CKPT_EVERY,
        async_upload=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        snapshot_ckpt_folder="/".join([BOTO_SAVE_PATH, "snapshot"]),
        oss_snapshot_freq=SNPASHOT_EVERY,
        stop_file_path=None,
        load_model_only_folder=None,
        load_given_ckpt=False,
        load_ckpt_folder=None,
        is_old_api=True,
    ),
    # Old interface format
    dict(
        enable_save_ckpt=True,
        save_ckpt_folder=LOCAL_SAVE_PATH,
        load_optimizer=True,
        checkpoint_every=CKPT_EVERY,
        async_upload=False,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        snapshot_ckpt_folder="/".join([LOCAL_SAVE_PATH, "snapshot"]),
        oss_snapshot_freq=SNPASHOT_EVERY,
        stop_file_path=None,
        load_model_only_folder=None,
        load_given_ckpt=False,
        load_ckpt_folder=None,
        is_old_api=True,
    ),
    # New interface format
    dict(
        enable_save_ckpt=True,
        save_ckpt_folder=BOTO_SAVE_PATH,
        checkpoint_every=CKPT_EVERY,
        async_upload=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        oss_snapshot_freq=SNPASHOT_EVERY,
        stop_file_path=None,
        is_old_api=False,
        auto_resume_latest_ckpt=True,
    ),
    dict(
        enable_save_ckpt=True,
        save_ckpt_folder=LOCAL_SAVE_PATH,
        checkpoint_every=CKPT_EVERY,
        async_upload=False,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        oss_snapshot_freq=SNPASHOT_EVERY,
        stop_file_path=None,
        load_ckpt_folder=None,
        is_old_api=False,
        auto_resume_latest_ckpt=True,
    ),
]


def overwrite_optim_state(optim, set_value):
    if isinstance(optim, HybridZeroOptimizer):
        for group_id, p in optim._fp32_flat_param_groups_of_current_rank.items():
            if optim._zero_local_rank not in optim.param_group_no_params_ranks[group_id]:
                # p.copy_(torch.full_like(p, set_value, dtype=p.dtype))
                p.data.fill_(set_value)
        for group_id in range(len(optim._fp16_param_groups)):
            if optim._zero_local_rank not in optim.param_group_no_params_ranks[group_id]:
                fp16_p = optim._param_store.get_flat_fp16_param_by_rank_group(
                    rank=optim._zero_local_rank, group_id=group_id
                )
                fp16_p.fill_(set_value)
    else:
        for group in optim.param_groups:
            for p in group["params"]:
                # p.copy_(torch.full_like(p, set_value, dtype=p.dtype))
                p.data.fill_(set_value)


def compare_optim_state(optim1, optim2):
    re = True
    if isinstance(optim1, HybridZeroOptimizer):
        fp32_buff1 = optim1._fp32_flat_param_groups_of_current_rank
        fp32_buff2 = optim2._fp32_flat_param_groups_of_current_rank
        for group_id_1, group_id_2 in zip(fp32_buff1, fp32_buff2):
            re &= group_id_1 == group_id_2
            if optim1.zero_local_rank not in optim1.param_group_no_params_ranks[group_id_1]:
                re &= torch.equal(fp32_buff1[group_id_1], fp32_buff1[group_id_2])
    else:
        for group1, group2 in zip(optim1.param_groups, optim2.param_groups):
            for p1, p2 in zip(group1["params"], group2["params"]):
                re &= torch.equal(p1, p2)
    return re


def compare_optim_value(optim, value):
    re = True
    if isinstance(optim, HybridZeroOptimizer):
        for group_id, p in optim._fp32_flat_param_groups_of_current_rank.items():
            if optim._zero_local_rank not in optim.param_group_no_params_ranks[group_id]:
                re &= torch.equal(p, torch.full_like(p, value, dtype=p.dtype))
        for group_id in range(len(optim._fp16_param_groups)):
            if optim._zero_local_rank not in optim.param_group_no_params_ranks[group_id]:
                fp16_p = optim._param_store.get_flat_fp16_param_by_rank_group(
                    rank=optim._zero_local_rank, group_id=group_id
                )
                re &= torch.equal(fp16_p, torch.full_like(fp16_p, value, dtype=fp16_p.dtype))
    else:
        for group in optim.param_groups:
            for p in group["params"]:
                re &= torch.equal(p, torch.full_like(p, value, dtype=p.dtype))
    return re


def overwrite_model_value(model, value):
    for p in model.parameters():
        # p.copy_(torch.full_like(p, value, dtype=p.dtype))
        p.data.fill_(value)


def compare_model_value(model, value):
    re = True
    for p in model.parameters():
        re &= torch.equal(p, torch.full_like(p, value, dtype=p.dtype))
    return re


@pytest.fixture(scope="function")
def del_tmp():
    del_tmp_file()
    yield
    del_tmp_file()


@pytest.mark.usefixtures("del_tmp")
@pytest.mark.usefixtures("reset_singletons")
@pytest.mark.parametrize("ckpt_config", ckpt_config_list)
def test_ckpt_mm(ckpt_config, init_dist_and_model):  # noqa # pylint: disable=unused-import

    ckpt_config = Config(ckpt_config)
    assert ckpt_config.checkpoint_every < TOTAL_STEP
    assert ckpt_config.oss_snapshot_freq < TOTAL_STEP

    model, opim = init_dist_and_model
    train_state = TrainState(gpc.config, None)
    init_storage_manager(ckpt_config)
    if isinstance(opim, HybridZeroOptimizer):
        print("Is HybridZeroOptimizer!", flush=True)
    else:
        print("Is naive Adam!", flush=True)

    ckpt_mm = CheckpointManager(ckpt_config, model=model, optimizer=opim)
    latest_ckpt_step = None
    for i in range(TOTAL_STEP + 1):
        overwrite_model_value(model, i)
        overwrite_optim_state(opim, i)

        train_state.batch_count = i
        train_state.step_count += 1

        if ckpt_mm.is_now_to_save_ckpt(train_state):
            latest_ckpt_step = i

        ckpt_mm.try_save_checkpoint(train_state)

    wait_async_upload_finish()
    latest_ckpt_info = ckpt_mm.query_lastest_ckpt()
    assert latest_ckpt_info is not None
    latest_ckpt = latest_ckpt_info["path"]
    if ckpt_mm.save_ckpt_folder.startswith("local"):
        assert latest_ckpt == "local:local_ckpt/snapshot/0", latest_ckpt
    else:
        assert latest_ckpt == f"{BOTO_SAVE_PATH}/snapshot/0", latest_ckpt

    del ckpt_mm
    ckpt_mm = CheckpointManager(ckpt_config, model=model, optimizer=opim)
    ckpt_mm.try_resume_training(train_state)
    assert latest_ckpt_step == 6
    assert train_state.step_count == 6
    assert train_state.batch_count == 6
    assert compare_optim_value(ckpt_mm.optimizer, latest_ckpt_step - 1), ckpt_mm.optimizer.param_groups[0]["params"][0]
    assert compare_model_value(ckpt_mm.model, latest_ckpt_step - 1), list(ckpt_mm.model.parameters())[0][0]

    if ckpt_mm.save_ckpt_folder.startswith("local:"):
        ckpt_mm.load_ckpt_info = dict(
            path=os.path.join(LOCAL_SAVE_PATH, "4"), content=["model", "sampler", "optimizer"], ckpt_type="normal"
        )
    else:
        ckpt_mm.load_ckpt_info = dict(
            path=os.path.join(BOTO_SAVE_PATH, "4"), content=["model", "sampler", "optimizer"], ckpt_type="normal"
        )

    ckpt_mm.try_resume_training(train_state)

    assert train_state.step_count == 4
    assert train_state.batch_count == 4
    assert compare_optim_value(ckpt_mm.optimizer, 3), ckpt_mm.optimizer.param_groups[0]["params"][0]
    assert compare_model_value(ckpt_mm.model, 3), list(ckpt_mm.model.parameters())[0][0]


@pytest.mark.usefixtures("del_tmp")
@pytest.mark.usefixtures("reset_singletons")
@pytest.mark.parametrize("ckpt_config", ckpt_config_list)
def test_ckpt_mm_ping(ckpt_config, init_dist_and_model):  # noqa # pylint: disable=unused-import
    ckpt_config = Config(ckpt_config)
    init_storage_manager(ckpt_config)

    model, opim = init_dist_and_model
    ckpt_mm = CheckpointManager(ckpt_config, model=model, optimizer=opim)
    ckpt_mm.try_ping_storage()


if __name__ == "__main__":
    pytest.main()
