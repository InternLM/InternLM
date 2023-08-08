import signal


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

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
