import json
import math
import os
import re
import time
from typing import Dict

import requests

from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


def initialize_light_monitor(monitor_address: str = None):
    """
    Initialize the lightweight monitoring module.

    Args:
        monitor_address (str, optional): The address of the monitor. Defaults to 'MONITOR_SERVER' environment.

    Raises:
        Exception: If any exceptions occur during initialization, they will be caught and logged as warnings.

    Example:
        Initialize the monitoring module with the default address
        ``initialize_light_monitor()``
    """
    try:
        from uniscale_monitoring import init_monitor

        init_monitor(monitor_address, is_root_rank=gpc.is_rank_for_log())
    except Exception as e:
        logger.warning(f"init monitor meet error: {e}")


def send_heartbeat(msg_type: str, msg: Dict):
    """
    Send a heartbeat message to a monitoring server.

    Args:
        msg_type (str): The type of heartbeat message, e.g., "train_metrics", "init_time", "stage_time".
        msg (Dict): A dictionary containing message data to be included in the heartbeat.

    Example:
        Sending a heartbeat message for training metrics
        ``send_heartbeat("train_metrics", {"loss": 0.1, "accuracy": 0.95})``

        Sending a heartbeat message for initialization time
        ``send_heartbeat("init_time", {"import_time": 0.25})``

        Sending a heartbeat message for stage time
        ``send_heartbeat("stage_time", {"fwd_time": 2.3, "bwd_time": 6.2})``
    """

    def nan2none(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    try:
        from uniscale_monitoring import send_meta

        data = {}
        for k, v in msg.items():
            if isinstance(v, Dict):
                for k1, v1 in v.items():
                    new_k = f"{k}_{k1}".split(" ")[0]
                    new_k = re.sub(r"[^a-zA-Z0-9_]", "_", new_k)
                    data[new_k] = nan2none(v1)
            else:
                new_k = k.split(" ")[0]
                new_k = re.sub(r"[^a-zA-Z0-9_]", "_", new_k)
                data[new_k] = nan2none(v)

        if os.getenv("CLUSTER_NAME"):
            data.update({"cluster": os.getenv("CLUSTER_NAME")})
        if msg_type == "train_metrics":
            data.update({"msg_type": "train_metrics"})
        elif msg_type == "init_time":
            data.update({"msg_type": "init_time"})
        elif msg_type == "stage_time":
            data.update({"msg_type": "stage_time"})
        send_meta(data, timeout=0.1)
    except Exception as e:
        logger.warning(f"send heartbeat meet error: {e}")


def send_feishu_msg_with_webhook(webhook: str, title: str, message: str):
    """
    Use Feishu robot to send messages with the given webhook.

    Args:
        webhook (str): The webhook to be used to send message.
        title (str): The message title.
        message (str): The message body.

    Returns:
        The response from the request. Or catch the exception and return None.

    Raises:
        Exception: An exception rasied by the HTTP post request.

    """

    headers = {"Content-Type": "application/json;charset=utf-8"}
    msg_body = {
        "timestamp": int(time.time()),
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [
                        [
                            {
                                "tag": "text",
                                "text": message,
                            },
                        ],
                    ],
                },
            },
        },
    }

    try:
        res = requests.post(webhook, data=json.dumps(msg_body), headers=headers, timeout=30)
        res = res.json()
        print(f"Feishu webhook response: {res}")
    except Exception as err:  # pylint: disable=W0703
        print(f"HTTP Post error: {err}")
        res = None

    return res
