import os

import pytest
import torch

from internlm.core.context.parallel_context import Config
from internlm.initialize.launch import get_config_value
from tests.test_utils.common_fixture import (  # noqa # pylint: disable=unused-import
    ASYNC_TMP_FOLDER,
    BOTO_SAVE_PATH,
    LOCAL_SAVE_PATH,
    del_tmp_file,
    init_dist_and_model,
    reset_singletons,
)

ASYNC_TMP_FOLDER = "./async_tmp_folder"
ckpt_config_list = [
    # async boto
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        async_upload=True,
        save_folder=BOTO_SAVE_PATH,
        test_id=0,
    ),
    # sync local
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=None,
        async_upload=False,
        save_folder=LOCAL_SAVE_PATH,
        test_id=1,
    ),
    # sync boto
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=None,
        async_upload=False,
        save_folder=BOTO_SAVE_PATH,
        test_id=2,
    ),
    # async local
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        async_upload=True,
        save_folder=LOCAL_SAVE_PATH,
        test_id=3,
    ),
]


@pytest.fixture(scope="function")
def del_tmp():
    del_tmp_file()
    yield
    del_tmp_file()


@pytest.mark.usefixtures("del_tmp")
@pytest.mark.usefixtures("reset_singletons")
@pytest.mark.parametrize("ckpt_config", ckpt_config_list)
def test_storage_mm_save_load(ckpt_config, init_dist_and_model):  # noqa # pylint: disable=unused-argument
    from internlm.utils.storage_manager import (
        check_folder,
        get_fns,
        init_storage_manager,
        llm_load,
        llm_save,
        wait_async_upload_finish,
    )

    ckpt_config = Config(ckpt_config)
    enable_save_ckpt = get_config_value(ckpt_config, "enable_save_ckpt", False)
    async_upload_tmp_folder = get_config_value(ckpt_config, "async_upload_tmp_folder", False)
    async_upload = get_config_value(ckpt_config, "async_upload", False)

    init_storage_manager(enable_save_ckpt, async_upload_tmp_folder, async_upload)

    tobj = torch.rand(64, 64)
    save_fn = os.path.join(ckpt_config.save_folder, "test.pt")
    llm_save(save_fn, tobj)
    if ckpt_config.test_id == 0:
        wait_async_upload_finish()
    check_folder(save_fn)
    assert get_fns(ckpt_config.save_folder)[0] == "test.pt"
    load_obj = llm_load(save_fn, map_location="cpu")
    assert 0 == ((load_obj != tobj).sum())
