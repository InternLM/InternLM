import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from internlm.core.context.parallel_context import Config
from internlm.core.trainer import TrainState
from internlm.solver.optimizer.hybrid_zero_optim import HybridZeroOptimizer
from internlm.utils.common import SingletonMeta
from internlm.utils.model_checkpoint import CheckpointManager
from internlm.utils.storage_manager import wait_async_upload_finish
from tests.test_utils.common_fixture import (  # noqa # pylint: disable=unused-import
    ASYNC_TMP_FOLDER,
    BOTO_SAVE_PATH,
    LOCAL_SAVE_PATH,
    del_tmp_file,
    init_config,
    init_dist_and_model,
    reset_singletons,
)

# (TOTAL_STEP, CKPT_EVERY, SNPASHOT_EVERY)
step_info_list = [(8, 4, 2), (3, 4, 2), (1, 6, 3)]
ckpt_config_list = [
    # Old interface format
    dict(
        enable_save_ckpt=True,
        save_ckpt_folder=BOTO_SAVE_PATH,
        load_optimizer=True,
        checkpoint_every=0,
        async_upload=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        snapshot_ckpt_folder="/".join([BOTO_SAVE_PATH, "snapshot"]),
        oss_snapshot_freq=0,
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
        checkpoint_every=0,
        async_upload=False,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        snapshot_ckpt_folder="/".join([LOCAL_SAVE_PATH, "snapshot"]),
        oss_snapshot_freq=0,
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
        checkpoint_every=0,
        async_upload=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        oss_snapshot_freq=0,
        stop_file_path=None,
        is_old_api=False,
        auto_resume=True,
    ),
    dict(
        enable_save_ckpt=True,
        save_ckpt_folder=LOCAL_SAVE_PATH,
        checkpoint_every=0,
        async_upload=False,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        oss_snapshot_freq=0,
        stop_file_path=None,
        load_ckpt_folder=None,
        is_old_api=False,
        auto_resume=True,
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


def return_prefix_path(save_ckpt_folder):
    if save_ckpt_folder.startswith("local:"):
        return LOCAL_SAVE_PATH
    else:
        return BOTO_SAVE_PATH


def return_latest_save_path(save_ckpt_folder, total_step, snapshot_freq, ckpt_freq):

    snapshot_latest_step, normal_latest_step = 0, 0
    snapshot_latest_count, normal_latest_count = 0, 0

    for i in range(total_step):
        if (i + 1) % ckpt_freq == 0:
            normal_latest_step = i + 1
            normal_latest_count += 1
        else:
            if (i + 1) % snapshot_freq == 0:
                snapshot_latest_step = i + 1
                snapshot_latest_count += 1

    if snapshot_latest_step == 0:
        return None, None

    if normal_latest_step >= snapshot_latest_step:
        return normal_latest_step, os.path.join(return_prefix_path(save_ckpt_folder), f"{normal_latest_step}")
    elif normal_latest_step < snapshot_latest_step:
        if snapshot_latest_count % 2 == 0:
            re_path = f"{return_prefix_path(save_ckpt_folder)}/snapshot/0"
        else:
            re_path = f"{return_prefix_path(save_ckpt_folder)}/snapshot/1"
        return snapshot_latest_step, re_path
    else:
        assert False


@pytest.mark.usefixtures("del_tmp")
@pytest.mark.usefixtures("reset_singletons")
@pytest.mark.parametrize("step_info", step_info_list)
@pytest.mark.parametrize("ckpt_config", ckpt_config_list)
def test_ckpt_mm(step_info, ckpt_config, init_dist_and_model):  # noqa # pylint: disable=unused-import
    from internlm.core.context import global_context as gpc
    from internlm.utils.model_checkpoint import CheckpointLoadMask, CheckpointLoadType

    ckpt_config = Config(ckpt_config)
    total_step, checkpoint_every, oss_snapshot_freq = step_info
    print(total_step, checkpoint_every, oss_snapshot_freq, flush=True)
    ckpt_config.checkpoint_every = checkpoint_every
    ckpt_config.oss_snapshot_freq = oss_snapshot_freq

    bond_return_latest_save_path = partial(
        return_latest_save_path,
        ckpt_config.save_ckpt_folder,
        total_step,
        ckpt_config.oss_snapshot_freq,
        ckpt_config.checkpoint_every,
    )

    model, opim = init_dist_and_model
    train_state = TrainState(gpc.config, None)
    if isinstance(opim, HybridZeroOptimizer):
        print("Is HybridZeroOptimizer!", flush=True)
    else:
        print("Is naive Adam!", flush=True)

    ckpt_mm = CheckpointManager(ckpt_config, model=model, optimizer=opim)
    latest_ckpt_step = None
    for i in range(total_step):
        overwrite_model_value(model, i)
        overwrite_optim_state(opim, i)

        train_state.batch_count = i
        train_state.step_count += 1

        save_ckpts, _, _ = ckpt_mm.is_now_to_save_ckpt(train_state)
        if save_ckpts:
            latest_ckpt_step = i

        ckpt_mm.try_save_checkpoint(train_state)

    wait_async_upload_finish()
    latest_ckpt_info = ckpt_mm.query_lastest_ckpt()
    step, path = bond_return_latest_save_path()
    assert latest_ckpt_info["path"] == path
    if latest_ckpt_step is None:
        assert latest_ckpt_step == step
    else:
        assert latest_ckpt_step == step - 1

    # resume from before save skpt
    del ckpt_mm
    SingletonMeta._instances = {}
    ckpt_mm = CheckpointManager(ckpt_config, model=model, optimizer=opim)
    ckpt_mm.try_resume_training(train_state)

    if ckpt_config.checkpoint_every < total_step:
        # we use step_count to decide when save ckpt, os here latest_ckpt_step = step_count - 1
        assert train_state.step_count == latest_ckpt_step + 1
        assert train_state.batch_count == latest_ckpt_step + 1
        assert compare_optim_value(ckpt_mm.optimizer, latest_ckpt_step), ckpt_mm.optimizer.param_groups[0]["params"][0]
        assert compare_model_value(ckpt_mm.model, latest_ckpt_step), list(ckpt_mm.model.parameters())[0][0]

        if ckpt_mm.save_ckpt_folder.startswith("local:"):
            ckpt_mm.load_ckpt_info = dict(
                path=os.path.join(LOCAL_SAVE_PATH, f"{ckpt_config.checkpoint_every}"),
                content=CheckpointLoadMask(("all",)),
                ckpt_type=CheckpointLoadType.INTERNLM,
            )
        else:
            ckpt_mm.load_ckpt_info = dict(
                path=os.path.join(BOTO_SAVE_PATH, f"{ckpt_config.checkpoint_every}"),
                content=CheckpointLoadMask(("all",)),
                ckpt_type=CheckpointLoadType.INTERNLM,
            )

        ckpt_mm.try_resume_training(train_state)

        assert train_state.step_count == ckpt_config.checkpoint_every
        assert train_state.batch_count == ckpt_config.checkpoint_every
        # compare value is same with i.
        assert compare_optim_value(ckpt_mm.optimizer, ckpt_config.checkpoint_every - 1), ckpt_mm.optimizer.param_groups[
            0
        ]["params"][0]
        assert compare_model_value(ckpt_mm.model, ckpt_config.checkpoint_every - 1), list(ckpt_mm.model.parameters())[
            0
        ][0]
    else:
        pass


STOP_FILE_PATH = "./alter.log"


def query_quit_file(rank, world_size=2):
    from internlm.core.context import global_context as gpc
    from internlm.initialize import initialize_distributed_env
    from internlm.utils.model_checkpoint import CheckpointSaveType

    ckpt_config = Config(
        dict(
            enable_save_ckpt=True,
            save_ckpt_folder=BOTO_SAVE_PATH,
            load_optimizer=True,
            checkpoint_every=0,
            async_upload=True,
            async_upload_tmp_folder=ASYNC_TMP_FOLDER,
            snapshot_ckpt_folder="/".join([BOTO_SAVE_PATH, "snapshot"]),
            oss_snapshot_freq=0,
            stop_file_path=STOP_FILE_PATH,
            load_model_only_folder=None,
            load_given_ckpt=False,
            load_ckpt_folder=None,
            is_old_api=True,
        ),
    )

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12376"

    initialize_distributed_env(config=init_config, launcher="torch", master_port=12376, args_check=False)
    train_state = TrainState(init_config, None)
    ckpt_mm = CheckpointManager(ckpt_config, model=None, optimizer=None)
    if rank == 0:
        with open(STOP_FILE_PATH, "w+") as f:
            f.write("5")
    dist.barrier()
    for i in range(10):
        train_state.step_count = i
        now_break, now_save_ckpt, save_type = ckpt_mm.quit_signal_handler(train_state)
        print(
            f"step:{i}, rank:{rank}, now_break:{now_break}, now_save_ckpt:{now_save_ckpt}, save_type:{save_type}",
            flush=True,
        )
        if train_state.step_count == 5:
            assert now_break is True
            assert now_save_ckpt is True
            assert save_type is CheckpointSaveType.NORMAL_CHECKPOINT
    dist.barrier()
    gpc.destroy()


def test_quit_siganl_handler():  # noqa # pylint: disable=unused-import
    import multiprocessing
    from multiprocessing.pool import Pool

    world_size = 2
    with Pool(processes=world_size, context=multiprocessing.get_context("spawn")) as pool:
        items = [(0,), (1,)]
        for result in pool.starmap(query_quit_file, items):
            print(f"Got result: {result}", flush=True)

    os.remove(STOP_FILE_PATH)


if __name__ == "__main__":
    pytest.main()
