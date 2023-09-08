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

    def start(self, reset_all=True):
        """Start the timer."""
        # need to reset all timers in a new batch
        if self.name_ == "one-batch" and reset_all is True:
            megatron_timer.reset()

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
            self.start(reset_all=False)
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}
        self.hist = {}
        self.names = []
        self.times = []

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def store_last_timers(self):
        """Store timers to two list"""
        self.names = []
        self.times = []
        for key, value in self.timers.items():
            senconds = round(float(value.elapsed(reset=False)), 4)
            self.names.append(key)
            self.times.append(senconds)
            if key not in self.hist:
                self.hist[key] = []
            self.hist[key].append(senconds)
            if len(self.hist[key]) > 10:
                self.hist[key].pop(0)

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

    def log(self, names, logger, normalizer=1.0, reset=True):
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

    def debug(self, names, logger, normalizer=1.0, reset=True):
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


megatron_timer = Timers()
