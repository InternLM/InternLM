#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time

import torch


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self.stream = torch.cuda.current_stream()

    def start(self):
        """Start the timer."""
        assert not self.started_, "timer has already been started"
        self.stream.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        self.stream.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            if name in self.timers:
                value = self.timers[name].elapsed(reset=reset) / normalizer
                writer.add_scalar(f"time/{name}-time", value, iteration)

    def log(self, names, logger, normalizer=1.0, reset=False):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = ""
        for name in names:
            if name in self.timers:
                elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
                string += " | {}: {:.2f}".format(name, elapsed_time)
        if not len(string):  # pylint: disable=C1802
            return
        string = "time (ms)" + string

        logger.info(string)
        return string

    def debug(self, names, logger, normalizer=1.0, reset=False):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = ""
        for name in names:
            if name in self.timers:
                elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
                string += " | {}: {:.2f}".format(name, elapsed_time)
        if not len(string):  # pylint: disable=C1802
            return
        string = "time (ms)" + string

        logger.debug(string)
        return string

    def reset(self):
        for _, t in self.timers.items():
            t.reset()

    def get_all_timer_results(self, normalizer=1.0, reset=False):
        assert normalizer > 0.0
        metric_time_dict = {}
        for name, tt in self.timers.items():
            metric_time_dict.update({name: tt.elapsed(reset=reset) * 1000.0 / normalizer})
        return sorted(metric_time_dict.items(), key=lambda x: x[0])  # sort base on value.


megatron_timer = Timers()
