#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch


class Beta2Scheduler:
    """
    Beta2Scheduler
    """

    def __init__(self, optimizer: torch.optim.Adam, init_beta2, c=0.8, cur_iter=-1):
        self.cur_iter = 0 if cur_iter == -1 else cur_iter
        self.init_beta2 = init_beta2
        self.c = c
        self.optimizer = optimizer
        assert isinstance(
            optimizer, (torch.optim.Adam, torch.optim.AdamW)
        ), "should use Adam optimzier, which has beta2"

    def step(self, cur_iter=None):
        if cur_iter is None:
            self.cur_iter += 1
        else:
            self.cur_iter = cur_iter

        new_beta2 = self.get_beta2()
        for pg in self.optimizer.param_groups:
            beta1, _ = pg["betas"]
            pg["betas"] = (beta1, new_beta2)

    def get_beta2(self):
        if self.c <= 0:
            return self.init_beta2
        scale = 1 - (1 / self.cur_iter**self.c)
        return max(self.init_beta2, scale)
