import os
from datetime import datetime


def now_time():
    return datetime.now().strftime("%b%d_%H-%M-%S")


def set_env_var(key, value):
    os.environ[str(key)] = str(value)


def get_job_id():
    job_id = "none"
    if os.getenv("SLURM_JOB_ID") is not None:
        job_id = os.getenv("SLURM_JOB_ID")
    else:
        job_id = os.getenv("K8S_WORKSPACE_ID")

    return job_id


def get_job_name():
    job_name = f"unknown-{now_time()}"
    if os.getenv("JOB_NAME") is not None:
        job_name = os.getenv("JOB_NAME")

    return job_name


def get_job_key():
    return f"{get_job_id()}_{get_job_name()}"


def get_process_rank():
    proc_rank = -1
    if os.getenv("SLURM_PROCID") is not None:
        proc_rank = int(os.getenv("SLURM_PROCID"))
    elif os.getenv("RANK") is not None:
        # In k8s env, we use $RANK.
        proc_rank = int(os.getenv("RANK"))

    assert proc_rank != -1, "get_process_rank can not get right process rank!"
    return proc_rank


def get_world_size():
    # We do not use torch's interface get_world_size to obtain the worldsize to prevent
    # errors when the init_process_group is not called.
    if os.getenv("SLURM_NTASKS") is not None:
        ntasks = int(os.environ["SLURM_NTASKS"])
    elif os.getenv("WORLD_SIZE") is not None:
        # In k8s env, we use $WORLD_SIZE.
        ntasks = int(os.enviro["WORLD_SIZE"])
    else:
        ntasks = 1

    return ntasks
