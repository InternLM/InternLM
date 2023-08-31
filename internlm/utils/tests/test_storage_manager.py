import pytest

from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config
from internlm.utils.tests.common_fixture import (  # noqa # pylint: disable=unused-import
    BOTO_SAVE_PATH,
    TOTAL_STEP,
    ckpt_config_list,
    del_tmp_file,
    init_dist_and_model,
    reset_singletons,
)


@pytest.mark.usefixtures("reset_singletons")
@pytest.mark.parametrize("ckpt_config", ckpt_config_list)
def test_storage_mm(ckpt_config, init_dist_and_model):  # noqa # pylint: disable=unused-import
    from internlm.utils.storage_manager import get_storage_manager, init_storage_manager

    ckpt_config = Config(ckpt_config)
    init_storage_manager(ckpt_config)
    s_mm = get_storage_manager()
