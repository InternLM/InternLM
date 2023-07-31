#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json

from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in "optimizer"}
        if isinstance(state_dict["after_scheduler"], (_LRScheduler, _CosineAnnealingLR)):
            state_dict["after_scheduler_type"] = type(state_dict["after_scheduler"]).__name__
            state_dict["after_scheduler_dict"] = state_dict["after_scheduler"].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        # state_dict = {key: value for key, value in self.__dict__.items() if key not in 'optimizer'}
        for key in list(self.__dict__.keys()):
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
        if isinstance(self.after_scheduler, (_LRScheduler, _CosineAnnealingLR)):
            assert type(self.after_scheduler).__name__ == state_dict["after_scheduler_type"]
            # state_dict['after_scheduler_dict'] = state_dict['after_scheduler'].state_dict()
            self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
            # del state_dict['after_scheduler']
        else:
            raise NotImplementedError()
        return state_dict

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingWarmupLR(WarmupScheduler):
    """Cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be applied.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0, eta_min: float = 0.0, last_epoch: int = -1):
        base_scheduler = _CosineAnnealingLR(
            optimizer, total_steps - warmup_steps, eta_min=eta_min, last_epoch=last_epoch
        )
        super().__init__(optimizer, warmup_steps, base_scheduler)


class FineTuneCosineAnnealingWarmupLR(CosineAnnealingWarmupLR):
    """
    FineTune Cosine Annealing Warmup LR.

    Args:
        optimizer: The optimizer object.
        total_steps (int): The number of total steps.
        init_steps (int): The number of init steps, default is 0.
        warmup_steps (int): The number of warm up steps, default is 0.
        eta_min (float): The minimum learning rate, default is 0.0.
        last_epoch: Last epoch, default is -1.

    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        init_steps: int = 0,
        warmup_ratio: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self._init_steps = init_steps
        self._warmup_steps = int(total_steps * warmup_ratio)
        # Use this value to calculate the lr of warmup, because warmup_epochs = init_steps + warmup_steps
        super().__init__(optimizer, total_steps, self._warmup_steps + init_steps, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:  # pylint: disable=E0203
                # This True switch is to avoid warning when the warmup reaches the preset value switch
                self.after_scheduler._get_lr_called_within_step = True
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        elif self.last_epoch >= self._init_steps:
            return [(self.last_epoch + 1 - self._init_steps) / self._warmup_steps * lr for lr in self.base_lrs]
        else:
            return [0 for lr in self.base_lrs]

    def __str__(self):
        return json.dumps(self.state_dict(), indent=4, sort_keys=True)
