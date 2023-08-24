import json
import time

import requests


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
