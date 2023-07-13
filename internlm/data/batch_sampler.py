#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import random
from typing import Iterator, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

T_co = TypeVar("T_co", covariant=True)


class DataParallelSampler(Sampler):
    """A data sampler for distributed data parallelism.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The Dataset for sampling.
        shuffle (bool, optional): Whether to shuffle data, defaults to False.
        seed (int, optional): The random seed used for sampling, defaults to 0.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = gpc.get_world_size(ParallelMode.DATA)
        self.rank = gpc.get_local_rank(ParallelMode.DATA)
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # type: ignore[arg-type]
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas)
                / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

            # update for next epoch so that there is no need to call
            # set_epoch manually
            self.epoch += 1
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def get_dpsampler_dataloader(
    dataset,
    shuffle=False,
    seed=1024,
    add_sampler=True,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    **kwargs,
):
    r"""Set up a deterministic dataloader (also configure seed workers, samplers and whether shuffle or not)

    Note:
        When pipeline parallel is enabled, shuffle cannot be True as it will result in mismatch between input data
        on the 1st stage and label on the last stage.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()

    if add_sampler and gpc.is_initialized(ParallelMode.DATA) and gpc.get_world_size(ParallelMode.DATA) > 1:
        sampler = DataParallelSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    else:
        sampler = None

    # Deterministic dataloader
    def seed_worker():
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    if sampler is None:
        return DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )
    else:
        return DataLoader(
            dataset,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )


class StaticBatchSampler:
    """
    A static batch sampler that generates batches with a fixed micro-batch size.

    Args:
        num_samples (int): The total number of samples in the dataset.
        batch_size (int): The batch size for the current rank. Defaults to 192.
        rampup_batch_size (str): A string with three space-separated integers representing the
                                 starting batch size, the increment, and the number of steps between
                                 each increment. For example, "192 24 8" means that the batch size
                                 starts at 192 and increases by 24 every 8 steps. Defaults to
                                 "6 2 8", which corresponds to a batch size of 2 for the first 6 steps.
        micro_bsz (int): The micro-batch size. Defaults to 2.
        seed (int): The random seed for shuffling the indices. Defaults to 0.
        drop_last (bool): If True, drop the last incomplete batch. Currently only supports True. Defaults to True.
        data_rank (int): The rank of the current process in the data parallel group. Defaults to 0.
        data_world_size (int): The number of processes in the data parallel group. Defaults to 1.
    """

    def __init__(
        self,
        datasets,
        batch_size=192,
        rampup_batch_size="6 2 8",
        micro_bsz=2,
        seed=0,
        drop_last=True,
        data_rank=0,
        data_world_size=1,
    ):
        assert drop_last is True, "Currently only support drop last"
        if rampup_batch_size:
            # In the process increase to batch_size
            start_bsz, bsz_incre, incre_every = map(int, rampup_batch_size.split())
        else:
            start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1
        self.raw_rampup_batch_size = rampup_batch_size
        self.start_bsz = start_bsz
        self.bsz_incre = bsz_incre
        self.incre_every = incre_every
        if gpc.is_initialized(ParallelMode.PIPELINE):
            assert (
                batch_size - self.start_bsz
            ) % self.bsz_incre == 0, f"{batch_size} - {self.start_bsz} should be multiple of {self.bsz_incre}"
            assert batch_size % micro_bsz == 0, f"batch_size({batch_size}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.start_bsz % micro_bsz == 0
            ), f"start_bsz({self.start_bsz}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.bsz_incre % micro_bsz == 0
            ), f"bsz_incre({self.bsz_incre}) should be multiple of micro_bsz({micro_bsz})"

        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.batch_count = 0
        self.micro_bsz = micro_bsz
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.num_consumed_samples_in_epoch = 0
        self.datasets = datasets
        self.num_samples = sum([len(ds) for ds in datasets])

        self.get_indices()  # get data

    def get_indices(self, old_indices=None):
        if old_indices is not None:
            assert (
                len(old_indices) <= self.num_samples
            ), f"The checkpoint has {len(old_indices)} samples, \
while the new restart use less samples ({self.num_samples})"

        else:
            old_indices = np.array([])

        # indices includes len(old_indices) but not self.num_samples
        indices = np.arange(len(old_indices), self.num_samples)
        self.rng_state = self.rng.get_state()
        self.rng.shuffle(indices)
        # Need to consider drop_last
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: \
{rampup_samples*self.data_world_size} Vs. self.num_samples: {self.num_samples}"

            num_samples = (self.num_samples - rampup_samples * self.data_world_size) // (
                self.batch_size * self.data_world_size
            )
            num_samples = num_samples * self.batch_size * self.data_world_size + rampup_samples * self.data_world_size
        else:
            num_samples = self.num_samples // (self.batch_size * self.data_world_size)
            num_samples = num_samples * self.batch_size * self.data_world_size
        indices = np.concatenate([old_indices, indices]).astype(int)  # It needs to be spliced with the previous
        indices = indices[:num_samples]
        self.indices = indices
        assert len(self.indices) >= self.batch_size, "The number of samples should be larger than batch_size"
        self.num_consumed_samples_in_epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + self.epoch)

    def __len__(self):
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: {rampup_samples*self.data_world_size} \
Vs. self.num_samples: {self.num_samples}"

            num_batches = (self.num_samples - rampup_samples * self.data_world_size) // self.batch_size
            num_batches = num_batches // self.data_world_size + self.incre_every * ramp_steps
        else:
            num_batches = self.num_samples // self.batch_size // self.data_world_size

        return num_batches

    def __iter__(self):
        indices = self.indices[self.data_rank :: self.data_world_size]
        while self.num_consumed_samples_in_epoch < len(indices):
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)
            batch = indices[self.num_consumed_samples_in_epoch : self.num_consumed_samples_in_epoch + cur_batch_size]
            yield batch
            self.num_consumed_samples_in_epoch += len(batch)  # Consider multiple processes.
            self.batch_count += 1
        self.get_indices()  # get a new round

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "seed": self.seed,
            "data_world_size": self.data_world_size,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "batch_count": self.batch_count,  # The batch_count here is due to the existence of multiple processes,
            # the batch may be oversent, and it needs to be overwritten by the external batch_count
            "indices": self.indices,  # The sequence used to breakpoint retraining is the same as before
        }

        return states

    def load_state_dict(self, states):
        for name in ("data_world_size", "raw_rampup_batch_size", "seed"):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.rng.set_state(states["rng_state"])
        self.get_indices(old_indices=None)  # Regenerate indices based on random state
        self.epoch = states["epoch"]
        self.batch_count = states["batch_count"]
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]

    def copy(self):
        copy_sampler = StaticBatchSampler(
            self.datasets,
            self.batch_size,
            self.raw_rampup_batch_size,
            self.micro_bsz,
            self.seed,
            drop_last=True,
            data_rank=self.data_rank,
            data_world_size=self.data_world_size,
        )

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler
