#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context

from contextlib import contextmanager

import torch
import torch.cuda
from torch import Tensor

from .process_group_initializer import ParallelMode


class SeedManager:
    """This class is a manager of all random seeds involved in the system."""

    def __init__(self):
        self._current_mode = None
        self._seeds = {}
        self._seed_states = {}

    @property
    def current_mode(self):
        return self._current_mode

    @property
    def seeds(self):
        return self._seeds

    @property
    def seed_states(self):
        return self._seed_states

    def set_state(self, parallel_mode: ParallelMode, state: Tensor):
        """Sets the state of the seed manager for `parallel_mode`."""
        assert parallel_mode in self._seed_states, f"{parallel_mode} not found in seed manager"
        self._seed_states[parallel_mode] = state

    def set_mode(self, parallel_mode: ParallelMode):
        """Sets the current mode of the seed manager."""
        if self.current_mode:
            # save state for current mode
            self._seed_states[self._current_mode] = torch.cuda.get_rng_state()

        # set new state for new mode
        self._current_mode = parallel_mode
        torch.cuda.set_rng_state(self._seed_states[parallel_mode])

    def add_seed(self, parallel_mode: ParallelMode, seed: int, overwrite: bool = False):
        """Adds a seed to the seed manager for `parallel_mode`."""
        assert isinstance(parallel_mode, ParallelMode), "Invalid ParallelMode"
        if not overwrite:
            assert parallel_mode not in self._seed_states, f"Seed for {parallel_mode} exists"
        elif parallel_mode in self._seed_states:
            print(f"Warning: {parallel_mode} seed overwritten.", flush=True)

        current_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
        self._seed_states[parallel_mode] = torch.cuda.get_rng_state()
        self._seeds[parallel_mode] = seed
        torch.cuda.set_rng_state(current_state)

    def reset(self):
        self._current_mode = None
        self._seeds = {}
        self._seed_states = {}


_SEED_MANAGER = SeedManager()


def get_seeds():
    """Returns the seeds of the seed manager.
    Returns:
        dict: The seeds of the seed manager.
    """
    return _SEED_MANAGER.seeds


def get_states(copy=False):
    """Returns the seed states of the seed manager.
    Returns:
        dict: The seed states of the seed manager.
    """
    states = _SEED_MANAGER.seed_states
    if copy:
        new_states = dict()
        for parallel_mode, state in states.items():
            new_states[parallel_mode] = state.clone()
        return new_states
    else:
        return _SEED_MANAGER.seed_states


def get_current_mode():
    """Returns the current mode of the seed manager.
    Returns:
        :class:`torch.ByteTensor`: The current mode of the seed manager.
    """
    return _SEED_MANAGER.current_mode


def add_seed(parallel_mode: ParallelMode, seed: int, overwrite: bool = False):
    """Adds a seed to the seed manager for `parallel_mode`."""
    _SEED_MANAGER.add_seed(parallel_mode, seed, overwrite)


def set_mode(parallel_mode: ParallelMode):
    """Sets the current mode of the seed manager."""
    _SEED_MANAGER.set_mode(parallel_mode)


def set_seed_states(parallel_mode: ParallelMode, state: Tensor):
    """Sets the state of the seed manager for `parallel_mode`."""
    _SEED_MANAGER.set_state(parallel_mode, state)


def sync_states():
    current_mode = get_current_mode()
    current_states = torch.cuda.get_rng_state()
    set_seed_states(current_mode, current_states)


@contextmanager
def seed(parallel_mode: ParallelMode):
    """A context for seed switch"""
    current_mode = _SEED_MANAGER.current_mode
    try:
        yield _SEED_MANAGER.set_mode(parallel_mode)
    finally:
        _SEED_MANAGER.set_mode(current_mode)
