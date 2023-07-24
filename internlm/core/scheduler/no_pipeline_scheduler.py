#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

import inspect
from typing import Any, Callable, Iterable

import torch

from internlm.core.engine import Engine
from internlm.core.context.parallel_context import global_context as gpc
from internlm.utils.common import conditional_context

from .base_scheduler import BaseScheduler


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

        _data, _label = self._load_micro_batch(
            data=data, label=label, offset=self._grad_accum_offset, micro_bsz=self._grad_accum_batch_size
        )
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
        post_fn: Callable = None,
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
            post_fn (Callable, optional): Call back function after executing data forward output.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            output = self._call_engine(engine, data)

            if post_fn is not None:
                post_fn(output, label)

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
        post_fn: Callable = None,
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
            post_fn (Callable, optional): Call back function after executing data forward output.

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
            
            def convert_data(input_ids, cu_seqlens, max_lenth):
                '''
                input_ids: (1, packed_length)
                
                Return:
                output: (batch_size, max_length)
                '''
                if isinstance(cu_seqlens, list):
                    assert len(cu_seqlens) == 1
                    cu_seqlens = cu_seqlens[0]
                
                if cu_seqlens is not None:
                    cu_seqlens = cu_seqlens.squeeze(0)

                if isinstance(cu_seqlens, torch.Tensor):
                    num_sequence = cu_seqlens.shape[0] - 1
                else:
                    raise RuntimeError("The cu_seqlens should be list or torch.Tensor type")
                assert not num_sequence == 0
                # obtain the unpacked tensors
                
                # output = torch.zeros(num_sequence, max_lenth, device=input_ids.device, dtype=input_ids.dtype)
                tensor_list = []
                for i in range(num_sequence):
                    tmp_tensor = input_ids[0, cu_seqlens[i]:cu_seqlens[i + 1]]
                    tensor_list.append(tmp_tensor)
                    # seq_length = cu_seqlens[i + 1] - cu_seqlens[i]
                    # output[i, 0:seq_length] = input_ids[0, cu_seqlens[i]:cu_seqlens[i + 1]]
                
                from torch.nn.utils.rnn import pad_sequence
                output = pad_sequence(tensor_list, batch_first=True)
                return output
            
            with torch.no_grad():
                _data['input_ids'] = convert_data(_data['input_ids'], _data['cu_seqlens'], gpc.config.data.seq_len)
                _label = convert_data(_label, _data['cu_seqlens'], gpc.config.data.seq_len)

            _output, _loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, self._grad_accum_size, post_fn
            )

            if return_loss:
                loss += _loss
            if return_output_label:
                outputs.append(_output)
                labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        return outputs, labels, loss
