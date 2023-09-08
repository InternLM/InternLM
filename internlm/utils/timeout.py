import datetime
import os
import signal
import socket
import traceback
from functools import wraps

from internlm.utils.logger import get_logger

logger = get_logger(__file__)


class Timeout:
    """Timer to execute code

    Adapted from https://github.com/reasoning-machines/pal

    Args:
        seconds (float): The maximum seconds to execute code
        error_message (str)
    """

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, error_type, value, traceback):
        signal.alarm(0)


ENABLE_TIMEOUT = os.getenv("INTERNLM_ENABLE_TIMEOUT", None)


timeout_threshold_dict = {
    "initialize_distributed_env": 120,
    "nopp_forward_backward_step": 360,
    "initialize_model": 10,
    "initialize_optimizer": 20,
    "optim_step": 30,
    "get_train_data_loader": 600,
    "get_validation_data_loader": 60,
    "load_new_batch": 10,
    "record_current_batch_training_metrics": 10,
    "save_checkpoint": 1200,
    "interleaved_forward_backward_step": 600,
    "nointerleaved_forward_backward_step": 600,
}

if ENABLE_TIMEOUT is not None:
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    LLM_NCCL_TIMEOUT = datetime.timedelta(seconds=int(os.getenv("NCCL_TIMEOUT", str(60))))
else:
    timeout_threshold_dict = dict.fromkeys(timeout_threshold_dict.keys(), 0)
    LLM_NCCL_TIMEOUT = datetime.timedelta(seconds=1800)


def try_get_gpc_rank():
    try:
        from internlm.core.context import global_context as gpc

        rank = gpc.get_global_rank()
    except:  # noqa  # pylint: disable=bare-except
        rank = "unknown"

    return f"host-{socket.gethostname()}-rank-{rank}"


def llm_timeout(seconds=0, func_name=None):
    """timeout decorator, Note that this decorator cannot be reentrant,
    otherwise the signal will be reset.

    Args:
        seconds (int, optional): timeout threshold. Defaults to 300.
        func_name (str, optional): the func who is been waited to timeout.
    """

    def decorator(func):
        nonlocal func_name
        if func_name is None:
            func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutError

            nonlocal seconds
            seconds = timeout_threshold_dict.get(func_name, seconds)

            if seconds > 0:
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError as e:
                logger.error(f"TimeoutError at {try_get_gpc_rank()}: {func_name}\\n {traceback.format_exc()}")
                raise e
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator
