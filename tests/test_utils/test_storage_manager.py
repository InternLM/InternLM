import os

import pytest
import torch

from internlm.core.context.parallel_context import Config
from internlm.initialize.launch import get_config_value
from tests.test_utils.common_fixture import (  # noqa # pylint: disable=unused-import
    BOTO_SAVE_PATH,
    LOCAL_SAVE_PATH,
    VOLC_SAVE_PATH,
    del_tmp_file,
    reset_singletons,
)

ASYNC_TMP_FOLDER = "./async_tmp_folder"
ckpt_config_list = [
    # sync local
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=None,
        async_upload=False,
        save_folder=LOCAL_SAVE_PATH,
        test_id=1,
    ),
    # async local
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        async_upload=True,
        save_folder=LOCAL_SAVE_PATH,
        test_id=2,
    ),
    # async boto
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        async_upload=True,
        save_folder=BOTO_SAVE_PATH,
        test_id=3,
    ),
    # sync boto
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=None,
        async_upload=False,
        save_folder=BOTO_SAVE_PATH,
        test_id=4,
    ),
    # async volc
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=ASYNC_TMP_FOLDER,
        async_upload=True,
        save_folder=VOLC_SAVE_PATH,
        test_id=5,
    ),
    # sync volc
    dict(
        enable_save_ckpt=True,
        async_upload_tmp_folder=None,
        async_upload=False,
        save_folder=VOLC_SAVE_PATH,
        test_id=6,
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
def test_storage_mm_save_load(ckpt_config):  # noqa # pylint: disable=unused-argument
    from internlm.utils.storage_manager import (
        check_folder,
        get_fns,
        init_storage_manager,
        llm_load,
        llm_save,
        wait_async_upload_finish,
    )

    ckpt_config = Config(ckpt_config)
    if os.environ.get("OSS_BUCKET_NAME") is None:
        if ckpt_config.test_id > 2:
            print("Pass boto3 and volc", flush=True)
            return

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


internlm_ckpt_path = [
    ("local:/mnt/ckpt/", "local", "/mnt/ckpt/"),
    ("local:./ckpt/", "local", "./ckpt/"),
    ("boto3:s3://oss_bucket/", "boto3", "s3://oss_bucket/"),
    ("boto3:oss_bucket/", "boto3", "oss_bucket/"),
    ("/mnt/ckpt/", "local", "/mnt/ckpt/"),
    ("./ckpt/", "local", "./ckpt/"),
    ("s3://oss_bucket/", "boto3", "s3://oss_bucket/"),
    ("volc:vc://oss_bucket/", "volc", "vc://oss_bucket/"),
    ("volc:oss_bucket/", "volc", "oss_bucket/"),
    ("vc://oss_bucket/", "volc", "vc://oss_bucket/"),
]


@pytest.mark.parametrize("ckpt_path", internlm_ckpt_path)
def test_try_get_storage_backend(ckpt_path):
    from internlm.utils.storage_manager import try_get_storage_backend

    ipath, a_prefix, a_cut_path = ckpt_path
    b_prefix, b_cut_path = try_get_storage_backend(ipath)
    assert a_prefix == b_prefix, f"{a_prefix} == {b_prefix}"
    assert a_cut_path == b_cut_path, f"{a_cut_path} == {b_cut_path}"
