#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import torch

from internlm.core.engine import Engine
from internlm.utils.common import conditional_context


class BaseScheduler(ABC):
    """A basic helper class to control the process of training or evaluation.
    It mainly composes of forward_backward_step for gradient backward and
    optimizer_step for parameters update.
    For the convenience to enable FP16, we aggregate all codes that contain the
    control of FP16 in class schedule.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data and arranges
            them into data and label.
    """

    def __init__(self, data_process_func: Callable = None):
        self.data_process_func = data_process_func

    @abstractmethod
    def pre_processing(self, engine: Engine):
        """To perform actions before running the schedule.

        Args:
           engine (internlm.core.Engine): InternLM engine for training and inference.
        """
        pass

    @abstractmethod
    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool,
        return_loss: bool = True,
        return_output_label: bool = True,
    ):
        """The process function over a batch of dataset for training or evaluation.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
            forward_only (bool): If True, the process won't include backward.
            return_loss (bool, optional): If False, the loss won't be returned.
            return_output_label (bool, optional): If False, the output and label won't be returned.
        """
        pass

    @staticmethod
    def _call_engine(engine: Engine, inputs: Any):
        """Calls the engine with the given inputs.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            inputs (Any): The inputs to the engine, can be of type torch.Tensor, list, tuple, or dict.
        """
        if isinstance(inputs, torch.Tensor):
            return engine(inputs)
        elif isinstance(inputs, (list, tuple)):
            return engine(*inputs)
        elif isinstance(inputs, dict):
            return engine(**inputs)
        else:
            raise TypeError(
                f"Expected engine inputs to be of type torch.Tensor, list, tuple, or dict, but got {type(inputs)}"
            )

    @staticmethod
    def _call_engine_criterion(engine: Engine, outputs: Any, labels: Any):
        """Calls the engine's criterion with the given outputs and labels.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            outputs (Any): The outputs from the model, can be of type torch.Tensor, list, tuple, or dict.
            labels (Any): The labels for the outputs, can be of type torch.Tensor, list, tuple, or dict.
        """
        assert isinstance(
            outputs, (torch.Tensor, list, tuple, dict)
        ), f"Expect output of model is (torch.Tensor, list, tuple), got {type(outputs)}"
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if isinstance(labels, torch.Tensor):
            labels = (labels,)

        if isinstance(outputs, (tuple, list)) and isinstance(labels, (tuple, list)):
            return engine.criterion(*outputs, *labels)
        elif isinstance(outputs, (tuple, list)) and isinstance(labels, dict):
            return engine.criterion(*outputs, **labels)
        elif isinstance(outputs, dict) and isinstance(labels, dict):
            return engine.criterion(**outputs, **labels)
        elif isinstance(outputs, dict) and isinstance(labels, (list, tuple)):
            raise ValueError(f"Expected labels to be a dict when the model outputs are dict, but got {type(labels)}")
        else:
            raise TypeError(
                f"Expected model outputs and labels to be of type torch.Tensor ' \
                '(which is auto-converted to tuple), list, tuple, or dict, ' \
                'but got {type(outputs)} (model outputs) and {type(labels)} (labels)"
            )


class NonPipelineScheduler(BaseScheduler):
    """A helper schedule class for no pipeline parallelism running environment.
    During one process, it loads a batch of dataset and feeds it to the model.
    After getting the output and calculating the loss, it will use :meth:`step`
    to update the parameters if it is in training mode.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data
            and returns a tuple in the form of (data, label), and it will be executed in load_batch.
        gradient_accumulation_steps(int, optional): the steps of gradient accumulation, 1 for disable
            gradient accumulation.

    Example:
        # this shows an example of customized data_process_func
        def data_process_func(dataloader_output):
            item1, item2, item3 = dataloader_output
            data = (item1, item2)
            label = item3
            return data, label
    """

    def __init__(self, data_process_func: Callable = None, gradient_accumulation_size: int = 1):
        # check that non-pipeline schedule data process func only takes in one parameter
        # which is the batch data
        if data_process_func:
            sig = inspect.signature(data_process_func)
            assert len(sig.parameters) == 1, (
                "The data_process_func only takes in one parameter for NonPipelineSchedule, "
                "which is a tuple of tensors for the current batch, "
                "i.e. data_process_func(dataloader_output)."
            )

        self._grad_accum_size = gradient_accumulation_size
        self._grad_accum_batch_size = 1  # static batch size for flash attetion.
        self._grad_accum_offset = 0

        super().__init__(data_process_func)

    def pre_processing(self, engine: Engine):
        """Performs actions before running the schedule.

        Args:
           engine (internlm.core.Engine): InternLM engine for training and inference.
        """
        pass

    def _load_accum_batch(self, data: Any, label: Any):
        """Loads a batch of data and label for gradient accumulation.

        Args:
            data (Any): The data to be loaded.
            label (Any): The label to be loaded.
        """
        _data = {
            k: v[self._grad_accum_offset : self._grad_accum_offset + self._grad_accum_batch_size]
            for k, v in data.items()
        }
        _label = label[self._grad_accum_offset : self._grad_accum_offset + self._grad_accum_batch_size]

        self._grad_accum_offset += self._grad_accum_batch_size

        return _data, _label

    def _train_one_batch(
        self,
        data: Any,
        label: Any,
        engine: Engine,
        forward_only: bool = False,
        return_loss: bool = True,
        scale_loss: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            output = self._call_engine(engine, data)

            if return_loss:
                loss = self._call_engine_criterion(engine, output, label)
                loss /= scale_loss

        # backward
        if not forward_only:
            engine.backward(loss)

        if not return_loss:
            loss = None

        return output, loss

    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool = False,
        return_loss: bool = True,
        return_output_label: bool = True,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        batch_data, batch_size = engine.load_batch(data_iter)

        assert (
            batch_size == self._grad_accum_size
        ), f"batch_size:{batch_size} must be equal to gradient accumulation steps:{self._grad_accum_size}"

        if self.data_process_func:
            data, label = self.data_process_func(batch_data)
        else:
            # if not batch data process func is given,
            # then we regard the batch data as a simple tuple of (data, label)
            data, label = batch_data

        loss = 0 if return_loss else None
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if _current_accum_step == self._grad_accum_size - 1:
                engine.optimizer.skip_grad_reduce = False
            else:
                engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, self._grad_accum_size
            )

            if return_loss:
                loss += _loss

            outputs.append(_output)
            labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        return outputs, labels, loss
