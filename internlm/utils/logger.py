#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import os

LOGGER_NAME = "internlm"
LOGGER_FORMAT = "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s in %(funcName)s -- %(message)s"
LOGGER_LEVEL = "info"
LOGGER_LEVEL_CHOICES = ["debug", "info", "warning", "error", "critical"]
LOGGER_LEVEL_HELP = (
    "The logging level threshold, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info'"
)

uniscale_logger = None


def get_logger(logger_name: str = LOGGER_NAME, logging_level: str = LOGGER_LEVEL) -> logging.Logger:
    """Configure the logger that is used for uniscale framework.

    Args:
        logger_name (str): used to create or get the correspoding logger in
            getLogger call. It will be "internlm" by default.
        logging_level (str, optional): Logging level in string or logging enum.

    Returns:
        logger (logging.Logger): the created or modified logger.

    """

    if uniscale_logger is not None:
        return uniscale_logger

    logger = logging.getLogger(logger_name)

    if logging_level not in LOGGER_LEVEL_CHOICES:
        logging_level = LOGGER_LEVEL
        print(LOGGER_LEVEL_HELP)

    logging_level = logging.getLevelName(logging_level.upper())

    handler = logging.StreamHandler()
    handler.setLevel(logging_level)
    logger.setLevel(logging_level)
    handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
    logger.addHandler(handler)

    return logger


def initialize_uniscale_logger(
    job_name: str = None,
    launch_time: str = None,
    file_name: str = None,
    name: str = LOGGER_NAME,
    level: str = LOGGER_LEVEL,
    file_path: str = None,
    is_std: bool = True,
):
    """
    Initialize uniscale logger.

    Args:
        job_name (str): The name of training job, defaults to None.
        launch_time (str): The launch time of training job, defaults to None.
        file_name (str): The log file name, defaults to None.
        name (str): The logger name, defaults to "internlm".
        level (str): The log level, defaults to "info".
        file_path (str): The log file path, defaults to None.
        is_std (bool): Whether to output to console, defaults to True.

    Returns:
        Uniscale logger instance.
    """

    try:
        from uniscale_monitoring import get_logger as get_uniscale_logger
    except ImportError:
        print("Failed to import module uniscale_monitoring. Use default python logger.")
        return None

    if not file_path:
        assert (
            job_name and launch_time and file_name
        ), "If file_path is None, job_name, launch_time and file_name must be setted."
        log_file_name = file_name
        log_folder = os.path.join("RUN", job_name, launch_time, "logs")
        log_dir = os.path.join(log_folder, log_file_name)
        file_path = log_dir

    logger = get_uniscale_logger(name=name, level=level, filename=file_path, is_std=is_std)
    if isinstance(logger, (list, tuple)):
        logger = list(logger)[0]

    global uniscale_logger
    uniscale_logger = logger

    return logger
