#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

import json
from typing import Iterable, Optional

from internlm.core.engine import Engine
from internlm.core.scheduler import (
    BaseScheduler,
    InterleavedPipelineScheduler,
    NonPipelineScheduler,
    PipelineScheduler,
)


class TrainState:
    """
    The TrainState class is used to record the current state of training.

    Args:
        train_dl (DataLoader): The DataLoader object used for training.
    """

    def __init__(self, config) -> None:
        # The number of batches produced by the data iterator
        self.batch_count: int = 0
        # Used to store the number of samples consumed in the current epoch
        self.num_consumed_samples_in_epoch: int = 0
        # Total number of tokens consumed
        self.num_consumed_tokens: int = 0
        # Number of batches skipped due to inf or nan values
        self.inf_nan_skip_batches: int = 0
        # Records the number of updates, skipped batches and inf batches are not counted
        self.step_count: int = 0

        # Total step count
        self.total_steps: int = config.data.total_steps

    def init_batch_sampler(self, train_dl):
        # Copy of the batch sampler from the DataLoader
        self.batch_sampler = train_dl.batch_sampler.copy()
        # Iterator for the batch sampler
        self.batch_sampler_iter = iter(self.batch_sampler)

    def __str__(self) -> str:
        """Returns a string representation of the training state in JSON format."""
        info = {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "num_consumed_tokens": self.num_consumed_tokens,
            "inf_nan_skip_batches": self.inf_nan_skip_batches,
            "step_count": self.step_count,
        }

        return json.dumps(info, indent=4, sort_keys=True)

    def load_state_dict(self, other_stuffs, train_dl):
        """
        Resumes training from a checkpoint.

        Args:
            other_stuffs (dict): Other information needed to resume training.
            train_dl (DataLoader): The DataLoader object used for training.
        """

        self.batch_count = other_stuffs["batch_count"] + 1  # here you need to shift a batch backward
        self.num_consumed_samples_in_epoch = other_stuffs["num_consumed_samples_in_epoch"]
        self.num_consumed_tokens = other_stuffs["num_consumed_tokens"]
        self.inf_nan_skip_batches = other_stuffs["inf_nan_skip_batches"]
        # compatible with previous checkpoints without this parameter
        self.step_count = other_stuffs.get("step_count", other_stuffs["batch_count"]) + 1

        # track the actual updates of sampler when using weighted sampling
        self.batch_sampler = train_dl.batch_sampler.copy()
        self.batch_sampler_iter = iter(self.batch_sampler)

    def state_dict(self):
        return {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "num_consumed_tokens": self.num_consumed_tokens,
            "inf_nan_skip_batches": self.inf_nan_skip_batches,
            "step_count": self.step_count,
        }


class Trainer:
    """This is a class tending for easy deployments of users' training and evaluation instead of
    writing their own scripts.

    Args:
        engine (:class:`Engine`): Engine responsible for the process function.
        schedule (:class:`BaseScheduler`, optional): Runtime schedule. Defaults to None.
    """

    def __init__(
        self,
        engine: Engine,
        schedule: Optional[BaseScheduler] = None,
    ):
        """Initializes the Trainer class.

        Args:
            engine (Engine): The engine responsible for the process function.
            schedule (Optional[BaseScheduler], optional): The runtime schedule. Defaults to None.
        """
        self._engine = engine

        # build schedule
        if schedule is None:
            self._schedule = NonPipelineScheduler()
        else:
            assert isinstance(
                schedule, BaseScheduler
            ), f"expected schedule to be of type BaseSchedule, but got {type(schedule)}"
            self._schedule = schedule

        if self.uses_pipeline:
            self._schedule.pre_processing(self._engine)

    @property
    def engine(self):
        return self._engine

    @property
    def schedule(self):
        return self._schedule

    @property
    def uses_pipeline(self):
        """Returns whether the pipeline parallel is used or not."""
        return isinstance(self._schedule, (PipelineScheduler, InterleavedPipelineScheduler))

    def train(self):
        self._engine.train()

    def eval(self):
        self._engine.eval()

    def zero_grad(self):
        self._engine.zero_grad()

    def step(self):
        return self._engine.step()

    def execute_schedule(self, data_iter: Iterable, **kwargs):
        """Runs the forward, loss computation, and backward for the model.
        Returns a tuple of (output, label, loss).

        Args:
            data_iter (Iterable): The data iterator.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss).
        """
        output, label, loss = self._schedule.forward_backward_step(self._engine, data_iter, **kwargs)
        return output, label, loss
