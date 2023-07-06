#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

from typing import List, Optional

import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler

from internlm.core.gradient_handler import BaseGradientHandler
from internlm.solver.beta2_scheduler import Beta2Scheduler
from internlm.solver.optimizer.hybrid_zero_optim import BaseOptimizer
from internlm.utils.common import get_batch_size, move_to_device


class Engine:
    """
    The Engine class is responsible for managing the training and evaluation process of a neural network model.
    It handles the forward and backward passes, parameter updates, gradient handling, and mode switching between
    training and evaluation.

    Args:
        model (torch.nn.Module): The neural network model to be trained or evaluated.
        optimizer (BaseOptimizer): The optimizer used for updating the parameters of the model.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler for the optimizer.
                                                                        Default is None.
        beta2_scheduler (internlm.solver.beta2_scheduler.Beta2Scheduler, optional): The beta2 scheduler for the
                                                                                    optimizer. Default is None.
        criterion (torch.nn.modules.loss._Loss, optional): The loss function used for calculating the loss during
                                                           training. Default is None.
        gradient_handlers (List[BaseGradientHandler], optional): A list of gradient handlers used in the backward pass.
                                                                 Default is None.
        clip_grad_norm (float, optional): The norm value for gradient clipping. Default is 0.0.

    Examples:
        >>> # define model, criterion, optimizer, lr_scheduler, train_dataloader for your training
        >>> model = ...
        >>> criterion = ...
        >>> optimizer = ...
        >>> train_dataloader = ...
        >>> engine, _, _, _ = internlm.initialize_engine(model, optimizer, criterion)
        >>> engine.train()
        >>> for inputs, labels in train_dataloader
        >>>     # set gradients to zero
        >>>     engine.zero_grad()
        >>>     # run forward pass
        >>>     outputs = engine(inputs)
        >>>     # compute loss value and run backward pass
        >>>     loss = engine.criterion(outputs, labels)
        >>>     engine.backward(loss)
        >>>     # update parameters
        >>>     engine.step()
    """

    def __init__(
        self,
        model: Module,
        optimizer: BaseOptimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        beta2_scheduler: Optional[Beta2Scheduler] = None,
        criterion: Optional[_Loss] = None,
        gradient_handlers: Optional[List[BaseGradientHandler]] = None,
        clip_grad_norm: float = 0.0,
    ):
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._beta2_scheduler = beta2_scheduler
        self._criterion = criterion
        self._clip_grad_norm = clip_grad_norm

        # state
        self.training = True  # default

        # build gradient handler
        self._gradient_handlers = gradient_handlers if gradient_handlers else []

    @property
    def model(self):
        """Returns the model attached to the engine."""
        return self._model

    @property
    def optimizer(self):
        """Returns the optimizer attached to the engine."""
        return self._optimizer

    @property
    def criterion(self):
        """Returns the criterion (loss function) attached to the engine."""
        return self._criterion

    def _all_reduce_gradients(self):
        """Handles all-reduce operations of gradients across different parallel groups."""
        for handler in self._gradient_handlers:
            handler.handle_gradient()

    def zero_grad(self):
        """Sets the gradient of all parameters in the model to zero."""
        self.optimizer.zero_grad()

    def step(self):
        """
        Executes the parameter update step. This includes all-reduce operations of gradients, gradient clipping,
        and parameter update. If successful, it also steps the learning rate scheduler and beta2 scheduler
        if they exist.

        Returns:
            success (bool): Whether the parameter update was successful.
            grad_norm (float): The norm of the gradient after clipping.
        """
        self._all_reduce_gradients()
        self.optimizer.clip_grad_norm(self.model, self._clip_grad_norm)

        success, grad_norm = self.optimizer.step()

        if success and self._lr_scheduler is not None:
            self._lr_scheduler.step()

        if success and self._beta2_scheduler is not None:
            self._beta2_scheduler.step()

        return success, grad_norm

    def train(self):
        """Sets the model to training mode."""
        self.training = True
        self._model.train()

    def eval(self):
        """Sets the model to evaluation mode."""
        self.training = False
        self._model.eval()

    def backward(self, loss: torch.Tensor):
        """
        Starts the backward propagation given the loss value computed by a loss function.

        Args:
            loss (torch.Tensor): The loss value computed by a loss function.
        """
        return self.optimizer.backward(loss)

    def backward_by_grad(self, tensor, grad):
        """
        Starts the backward propagation given the gradient of the output tensor.

        Args:
            tensor (torch.Tensor): The output tensor.
            grad (torch.Tensor): The gradient passed back to the output tensor.
        """
        return self.optimizer.backward_by_grad(tensor, grad)

    def __call__(self, *args, **kwargs):
        """
        Runs the forward step for the model.

        Returns:
            torch.Tensor: The output of the model.
        """
        return self.model(*args, **kwargs)

    def load_batch(self, data_iter, to_gpu=True):
        """
        Loads a batch from the data iterator. It returns the data and labels which are
        already in the same GPU as where the model is.

        Args:
            data_iter (Iterable): The data iterator from which to get a batch of data, obtained by calling
                                  iter(dataloader).
            to_gpu (bool, optional): Whether the data should be moved to the GPU. Default is True.

        Returns:
            Tuple (torch.Tensor, torch.Tensor): A tuple of (data, label).
        """
        if data_iter is None:
            raise RuntimeError("Dataloader is not defined.")
        try:
            batch_data = next(data_iter)
        except TypeError:
            batch_data = data_iter

        if to_gpu:
            batch_data = move_to_device(batch_data)
        batch_size = get_batch_size(batch_data)

        return batch_data, batch_size
