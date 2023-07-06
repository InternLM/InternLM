#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging

LOGGER_NAME = "internlm"
LOGGER_FORMAT = "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s in %(funcName)s -- %(message)s"
LOGGER_LEVEL = "info"
LOGGER_LEVEL_CHOICES = ["debug", "info", "warning", "error", "critical"]
LOGGER_LEVEL_HELP = (
    "The logging level threshold, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info'"
)


def get_logger(logger_name: str = LOGGER_NAME, logging_level: str = LOGGER_LEVEL) -> logging.Logger:
    """Configure the logger that is used for uniscale framework.

    Args:
        logger_name (str): used to create or get the correspoding logger in
            getLogger call. It will be "internlm" by default.
        logging_level (str, optional): Logging level in string or logging enum.

    Returns:
        logger (logging.Logger): the created or modified logger.

    """
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
