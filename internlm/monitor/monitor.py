import os
import signal
import socket
import time
from threading import Thread

from internlm.core.context import global_context as gpc
from internlm.monitor.alert import send_feishu_msg_with_webhook

from .utils import get_job_key, get_process_rank, set_env_var

ENABLE_MONITOR = False
JOB_NAME = None
MONITOR_THREAD = None
FEISHU_WEBHOOK_ADDRESS = None

# env var key
LOSS = "LOSS"
STEP_ID = "STEP_ID"
LAST_ACTIVE_TIMESTAMP = "LAST_ACTIVE_TIMESTAMP"

# the time seconds value of monitor interval
MONITOR_INTERVAL_SECONDS = 300
# the limit multiple of previous loss value
LOSS_SPIKE_LIMIT = 1.5
# loss value of last step
LAST_STEP_LOSS = -1


def send_alert_message(address: str = FEISHU_WEBHOOK_ADDRESS, title: str = None, message: str = None):
    if ENABLE_MONITOR:
        send_feishu_msg_with_webhook(
            webhook=address,
            title=title if title else get_job_key(),
            message=message,
        )


class MonitorTracker(Thread):
    """
    Track job status and alert to Feishu during job training.

    Args:
        alert_address (str): The Feishu webhook address for sending alerting messages.
        check_interval (float): The interval in seconds for monitoring checks. Defaults to MONITOR_INTERVAL_SECONDS.
        loss_spike_limit (float): The threshold for detecting loss value spikes. Defaults to LOSS_SPIKE_LIMIT.
    """

    def __init__(
        self,
        alert_address: str,
        check_interval: float = MONITOR_INTERVAL_SECONDS,
        loss_spike_limit: float = LOSS_SPIKE_LIMIT,
    ):
        super().__init__()
        self.alert_address = alert_address
        self.check_interval = check_interval
        self.loss_spike_limit = loss_spike_limit
        self.last_active_time = -1
        self.last_loss_value = -1
        self.stopped = False
        self.start()

    def run(self):
        """
        start the monitor tracker.
        """

        while not self.stopped:
            try:
                self._check_stuck()
                self._check_loss_spike()
            except Exception:
                continue
            time.sleep(self.check_interval)

    def _check_stuck(self):
        """
        Check training status for potential stuck condition.
        """

        new_active_time = -1
        if os.getenv(LAST_ACTIVE_TIMESTAMP) is not None:
            new_active_time = os.getenv(LAST_ACTIVE_TIMESTAMP)
        if int(new_active_time) <= int(self.last_active_time) and new_active_time != -1:
            self._send_alert("Training may be in stuck status, please check it.")
        self.last_active_time = new_active_time

    def _check_loss_spike(self):
        """
        Check for loss value spikes.
        """

        if gpc.is_rank_for_log():
            new_loss_value = -1
            new_step_id = -1
            if os.getenv(LOSS) is not None:
                new_loss_value = os.getenv(LOSS)
            if os.getenv(STEP_ID) is not None:
                new_step_id = os.getenv(STEP_ID)

            if (float(new_loss_value) / float(self.last_loss_value)) > self.loss_spike_limit and new_loss_value != -1:
                assert int(new_step_id) >= 0
                self._send_alert(
                    f"Checking periodically: Loss spike may be happened in step {new_step_id}, "
                    f"loss value from {self.last_loss_value} to {new_loss_value}, please check it."
                )

            self.last_loss_value = new_loss_value

    def _send_alert(self, message):
        """
        Send alerting message to the Feishu webhook address.

        Args:
            message (str): The alerting message to be sent.
        """

        if get_process_rank() == 0:
            send_alert_message(
                address=self.alert_address,
                message=message,
            )

    def stop(self):
        """
        Stop the monitor tracker.
        """

        self.stopped = True


def monitor_exception(excp_info: str):
    if ENABLE_MONITOR:
        filtered_trace = excp_info.split("\n")[-10:]
        format_trace = ""
        for line in filtered_trace:
            format_trace += "\n" + line
        send_alert_message(
            address=FEISHU_WEBHOOK_ADDRESS,
            message=f"Catch Exception from {socket.gethostname()} with proc id {get_process_rank()}:{format_trace}",
        )


def monitor_loss_spike(step_count, cur_step_loss):
    if ENABLE_MONITOR:
        set_env_var(key=LOSS, value=cur_step_loss)
        set_env_var(key=STEP_ID, value=step_count)

        global LAST_STEP_LOSS
        if LAST_STEP_LOSS != -1 and cur_step_loss > LOSS_SPIKE_LIMIT * LAST_STEP_LOSS:
            send_alert_message(
                address=FEISHU_WEBHOOK_ADDRESS,
                message=(
                    f"Checking step by step: Loss spike may be happened in step {step_count}, "
                    f"loss value from {LAST_STEP_LOSS} to {cur_step_loss}, please check it."
                ),
            )
        LAST_STEP_LOSS = cur_step_loss


def handle_sigterm(feishu_webhook_address: str = FEISHU_WEBHOOK_ADDRESS):
    def sigterm_handler(sys_signal, frame):
        print("receive frame: ", frame)
        print("receive signal: ", sys_signal)
        if get_process_rank() == 0:
            send_alert_message(
                address=feishu_webhook_address,
                message=f"Process received signal {signal} and exited.",
            )

    signal.signal(signal.SIGTERM, sigterm_handler)


def initialize_monitor(
    job_name: str,
    feishu_webhook_address: str,
    max_waiting_seconds: float = MONITOR_INTERVAL_SECONDS,
    loss_spike_limit: float = LOSS_SPIKE_LIMIT,
):
    """
    1. Initialize some variables for monitoring.
    2. Start a thread, periodically check the training status, if there is any
        abnormality, send an alarm to Feishu.
    3. Catch SIGTERM signal, and send alert message to Feishu.
    """

    global ENABLE_MONITOR, JOB_NAME, FEISHU_WEBHOOK_ADDRESS
    ENABLE_MONITOR = True
    JOB_NAME = job_name
    FEISHU_WEBHOOK_ADDRESS = feishu_webhook_address
    set_env_var(key="JOB_NAME", value=job_name)

    global MONITOR_THREAD
    MONITOR_THREAD = MonitorTracker(
        alert_address=feishu_webhook_address,
        check_interval=max_waiting_seconds,
        loss_spike_limit=loss_spike_limit,
    )

    handle_sigterm(feishu_webhook_address=feishu_webhook_address)


def stop_monitor():
    """
    1. Stop the monitor and alert thread.
    """

    if MONITOR_THREAD is not None:
        MONITOR_THREAD.stop()
