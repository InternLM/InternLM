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
    elif os.getenv("KUBERNETES_POD_NAME") is not None:
        job_id = os.getenv("KUBERNETES_POD_NAME").split("-")[0]
    elif os.getenv("MLP_TASK_INSTANCE_ID") is not None:
        job_id = os.getenv("MLP_TASK_ID")

    return job_id


def get_job_name():
    job_name = f"unknown-{now_time()}"
    if os.getenv("JOB_NAME") is not None:
        job_name = os.getenv("JOB_NAME")

    return job_name


def get_job_key():
    return f"{get_job_id()}_{get_job_name()}"
