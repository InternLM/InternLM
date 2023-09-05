import pytest

from internlm.core.context.parallel_context import Config
from internlm.initialize.launch import get_config_value
from tests.test_utils.common_fixture import (  # noqa # pylint: disable=unused-import
    BOTO_SAVE_PATH,
    TOTAL_STEP,
    ckpt_config_list,
    del_tmp_file,
    init_dist_and_model,
    reset_singletons,
)


@pytest.mark.usefixtures("reset_singletons")
@pytest.mark.parametrize("ckpt_config", ckpt_config_list)
def test_storage_mm(ckpt_config, init_dist_and_model):  # noqa # pylint: disable=unused-argument
    from internlm.utils.storage_manager import get_storage_manager, init_storage_manager

    ckpt_config = Config(ckpt_config)
    enable_save_ckpt = get_config_value(ckpt_config, "enable_save_ckpt", False)
    async_upload_tmp_folder = get_config_value(ckpt_config, "async_upload_tmp_folder", False)
    async_upload = get_config_value(ckpt_config, "async_upload", False)

    init_storage_manager(enable_save_ckpt, async_upload_tmp_folder, async_upload)
    get_storage_manager()
