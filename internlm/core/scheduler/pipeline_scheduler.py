#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Union

import torch.cuda

import internlm.core.communication as comm
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.core.naive_amp import NaiveAMPModel
from internlm.utils.common import get_current_device, move_to_device
from internlm.utils.logger import get_logger
from internlm.utils.timeout import llm_timeout

from .base_scheduler import BaseScheduler, SchedulerHook

logger = get_logger(__file__)


def get_tensor_shape():
    if hasattr(gpc.config, "TENSOR_SHAPE"):
        return gpc.config.TENSOR_SHAPE

    if not gpc.is_initialized(ParallelMode.PIPELINE):
        return None

    if hasattr(gpc.config, "SEQ_LEN") and hasattr(gpc.config.data, "micro_bsz") and hasattr(gpc.config, "HIDDEN_SIZE"):
        if gpc.config.model.use_flash_attn:
            if gpc.config.parallel.sequence_parallel:
                sequence_world_size = gpc.get_world_size(ParallelMode.TENSOR)
                tensor_shape = (
                    gpc.config.SEQ_LEN * gpc.config.data["micro_bsz"] // sequence_world_size,
                    gpc.config.HIDDEN_SIZE,
                )
            else:
                tensor_shape = (
                    gpc.config.SEQ_LEN * gpc.config.data["micro_bsz"],
                    gpc.config.HIDDEN_SIZE,
                )
        else:
            tensor_shape = (
                gpc.config.data["micro_bsz"],
                gpc.config.SEQ_LEN,
                gpc.config.HIDDEN_SIZE,
            )
        return tensor_shape
    else:
        return None


def pack_return_tensors(return_tensors):
    output, label = tuple(zip(*return_tensors))
    if isinstance(output[0], torch.Tensor):
        output = torch.cat(output, dim=0)
    elif isinstance(output[0], (list, tuple)):
        output = tuple(torch.cat(tensors, dim=0) for tensors in zip(*output))
    else:
        raise TypeError("Output of model must be tensor or list/tuple of tensors")
    if isinstance(label[0], torch.Tensor):
        label = torch.cat(label, dim=0)
    else:
        merged_label = {k: [] for k in label[0].keys()}
        for d in label:
            for k, v in d.items():
                merged_label[k].append(v)
        label = {k: torch.cat(v, dim=0) for k, v in merged_label.items()}
    return output, label


@contextmanager
def switch_virtual_pipeline_parallel_rank(rank):
    prev_rank = gpc.virtual_pipeline_parallel_rank
    try:
        gpc.set_virtual_pipeline_parallel_rank(rank)
        yield
    finally:
        gpc.set_virtual_pipeline_parallel_rank(prev_rank)


@contextmanager
def switch_optimizer_grad_sync_skip_mode(optimizer, skip: bool = True):
    prev_mode = optimizer.skip_grad_reduce
    try:
        optimizer.skip_grad_reduce = skip
        yield
    finally:
        optimizer.skip_grad_reduce = prev_mode


class PipelineScheduler(BaseScheduler):
    """
    A helper schedule class for pipeline parallelism running environment.
    It uses non-interleaved 1F1B strategy. Other properties are similar as
    :class:`NonPipelineSchedule`.

    Args:
        num_microbatches (int): The number of microbatches.
        dtype (torch.dtype): Type of data. torch.float by default.
        data_process_func (Callable, optional):
            The post processing function which receives a micro batch of data, and it will be executed
            in `load_micro_batch`.
        tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
        scatter_gather_tensors (bool, optional):
            If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
        scheduler_hooks (Optional[List[SchedulerHook]], optional): List of scheduler hooks.
    """

    def __init__(
        self,
        num_microbatches: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        assert num_microbatches > 0, f"expected num_microbatches to be larger then 1, but got {num_microbatches}"

        assert not isinstance(
            tensor_shape, int
        ), "tensor_shape type should be one of Union[torch.Size, List[int], Tuple[int]]."

        super().__init__(data_process_func=data_process_func)

        self.num_microbatches = num_microbatches
        self.dtype = dtype
        self._hooks = scheduler_hooks

        self._tensor_shape = (
            tensor_shape if tensor_shape is None or isinstance(tensor_shape, torch.Size) else torch.Size(tensor_shape)
        )

        self.scatter_gather_tensors = (
            scatter_gather_tensors
            and gpc.is_initialized(ParallelMode.TENSOR)
            and gpc.get_world_size(ParallelMode.TENSOR) > 1
        )

        if gpc.config.parallel.sequence_parallel:
            self.scatter_gather_tensors = False

        # cache for the batch data
        self.batch_data = None

    @property
    def tensor_shape(self) -> torch.Size:
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, tensor_shape: torch.Size):
        self._tensor_shape = tensor_shape

    def pre_processing(self, engine):
        types = set()

        for param in engine.model.parameters():
            types.add(param.dtype)
        assert len(types) == 1, f"Mixed types of parameter detected, {types}"

        self.dtype = types.pop()

    @staticmethod
    def _call_engine(engine, data):  # pylint: disable=W0237
        if data is None:
            return None

        if isinstance(data, torch.Tensor):
            return engine(data)
        elif isinstance(data, (list, tuple)):
            return engine(*data)
        elif isinstance(data, dict):
            stage_output = data.pop("stage_output", None)

            if stage_output is None:
                return engine(**data)
            elif isinstance(stage_output, torch.Tensor):
                return engine(stage_output, **data)
            elif isinstance(stage_output, (tuple, list)):
                return engine(*stage_output, **data)
            else:
                raise TypeError(
                    f"Expected stage_output to be of type torch.Tensor, list, or tuple, "
                    f"but got {type(stage_output)}"
                )
        else:
            raise TypeError(f"Expected data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")

    def load_batch(self, engine, data_iter):
        # Pipeline schedule just puts data in memory
        batch_data, batch_size = engine.load_batch(data_iter, to_gpu=False)
        assert batch_size % self.num_microbatches == 0, "Batch size should divided by the number of microbatches"

        self.microbatch_offset = 0
        self.batch_size = batch_size
        self.batch_data, self.batch_label = batch_data
        self.microbatch_size = self.batch_size // self.num_microbatches

    def load_micro_batch(self):
        micro_batch_data, micro_batch_label = self._load_micro_batch(
            data=self.batch_data, label=self.batch_label, offset=self.microbatch_offset, micro_bsz=self.microbatch_size
        )
        if self.data_process_func:
            micro_batch_data["input_ids"] = self.data_process_func(
                micro_batch_data["input_ids"], micro_batch_data["cu_seqlens"]
            )
            micro_batch_label = self.data_process_func(micro_batch_label, micro_batch_data["cu_seqlens"])

            micro_batch_data.pop("cu_seqlens")
            micro_batch_data.pop("indexes")

        micro_batch_data["label"] = micro_batch_label
        self.microbatch_offset += self.microbatch_size

        return move_to_device(micro_batch_data)

    def _get_data_label_for_current_step(self, stage_output, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            if gpc.is_first_rank(ParallelMode.PIPELINE):
                # for the first stage, we use the data from the
                # dataloader output by default
                data, label = micro_batch_data
            else:
                # for non-first stage, we use the output passed
                # by the previous as the model input
                data = stage_output
                _, label = micro_batch_data
        elif isinstance(micro_batch_data, dict):
            label = micro_batch_data.pop("label", None)
            data = {"stage_output": stage_output, **micro_batch_data}

        return data, label

    def _call_hooks(self, func_name: str, *args, **kwargs) -> None:
        for hook in self._hooks:
            getattr(hook, func_name)(self, *args, **kwargs)

    def _get_current_microbatch_id(self, step_id: int) -> int:
        """
        Get the current microbatch ID based on the step ID.
        In 1f1b scheduler, the microbatch ID is the same as the step ID,
        but it is important to note that the step ID is calculated separately
        for forward and backward passes.
        """
        return step_id

    def _forward_step(self, engine, input_obj, return_tensors, return_output_label=True, accum_loss=None):
        """
        Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Input tensor for this pipeline stage.
            return_tensors (List[:class:`torch.Tensor`]): A list of tensors to return.
            return_output_label (bool, optional): Whether returns output labels.
            accum_loss (optional): Where accumulated loss stores.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current
                pipeline stage.
        """
        micro_batch_data = self.load_micro_batch()
        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)

        self._call_hooks("before_forward", data)
        output_obj = self._call_engine(engine.model, data)
        self._call_hooks("after_forward", output_obj)

        if gpc.is_last_rank(ParallelMode.PIPELINE):
            self._call_hooks("post_helper_func", output_obj, label)
            if return_output_label:
                return_tensors.append((output_obj, label))
            if accum_loss is not None:
                self._call_hooks("before_criterion", output_obj, label)
                loss = self._call_engine_criterion(engine, output_obj, label)
                self._call_hooks("after_criterion", loss)

                loss_reduced = loss / self.num_microbatches
                accum_loss.add_(loss_reduced.detach())
                output_obj = loss_reduced

        return output_obj

    def _backward_step(self, engine, step_id, input_obj, output_obj, output_obj_grad):
        """
        Backward step through the passed-in output tensor. If it is the last stage, the
        output_obj_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            step_id (int): The ID of the current step.
            input_obj (Union[torch.Tensor, List[torch.Tensor]]): Input tensor for this stage.
            output_obj (Union[torch.Tensor, List[torch.Tensor]]): Output tensor for this stage.
            output_obj_grad (Union[torch.Tensor, List[torch.Tensor]]): Gradient of output tensor for this stage.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Gradient of input tensor.
        """

        # Retain the grad on the input_obj.
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj.retain_grad()
            else:
                for in_tensor in input_obj:
                    if in_tensor is not None:
                        in_tensor.retain_grad()

        # Backward pass.

        # Only the last microbatch does syncing grad.
        skip_grad_sync = self._get_current_microbatch_id(step_id) != self.num_microbatches - 1

        self._call_hooks("before_backward", output_obj, output_obj_grad)
        with switch_optimizer_grad_sync_skip_mode(engine.optimizer, skip_grad_sync):
            if output_obj_grad is None:
                engine.backward(output_obj)
            else:
                engine.backward_by_grad(output_obj, output_obj_grad)

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj_grad = input_obj.grad
            else:
                input_obj_grad = []
                for in_tensor in input_obj:
                    input_obj_grad.append(in_tensor.grad)
        self._call_hooks("after_backward", input_obj_grad)

        return input_obj_grad

    def _forward_only_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function performs forward only computation process. The scheduling of microbatches is similar to the
        warmup phase, where each microbatch first receives the forward input from the previous stage, then performs
        the forward computation, and finally passes the forward computation output to the next stage. There are two
        special cases to note:
        1. The first stage of the pipeline does not need to receive forward input; its input comes from the dataloader.
        2. The last stage of the pipeline does not need to send forward output; its output is returned to the user code
           for processing.

        Args:
            engine (colossalai.engine.Engine): internlm engine for training and inference.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
                output, label, and accumulated loss.
        """

        # Input, output tensors only need to be saved when doing backward passes
        return_tensors = []
        accum_loss = (
            torch.zeros(1, device=get_current_device())
            if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
            else None
        )

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        need_forward_meta = self.tensor_shape is None

        # Run all forward passes.
        for _ in range(self.num_microbatches):
            # Receive input from the previous stage
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

            # Perform forward computation
            output_obj = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
            )

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if need_forward_meta:
                    comm.send_obj_meta(output_obj)
                    need_forward_meta = False  # send only once.
                # Send the forward computation output to the next stage
                comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        return output, label, accum_loss

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        num_warmup_microsteps = (
            gpc.get_world_size(ParallelMode.PIPELINE) - gpc.get_local_rank(ParallelMode.PIPELINE) - 1
        )
        num_warmup_microsteps = min(num_warmup_microsteps, self.num_microbatches)
        num_1f1b_micropairs = self.num_microbatches - num_warmup_microsteps

        # Input, output tensors only need to be saved when doing backward passes
        input_objs = []
        output_objs = []
        return_tensors = []
        accum_loss = (
            torch.zeros(1, device=get_current_device())
            if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
            else None
        )

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        backward_recv_shapes = None
        need_forward_meta = self.tensor_shape is None

        # Run warmup forward passes.
        for i in range(num_warmup_microsteps):
            # Receive the input from the previous stage
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

            # Perform forward computation
            output_obj = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
            )

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if isinstance(output_obj, torch.Tensor):
                    backward_recv_shapes = output_obj.shape
                else:
                    backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]

                if need_forward_meta:
                    comm.send_obj_meta(output_obj)
                    need_forward_meta = False  # send only once.

            # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
            # forward computation
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

            input_objs.append(input_obj)
            output_objs.append(output_obj)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_1f1b_micropairs > 0:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta(forward_recv_shapes)
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

        # Run 1F1B in steady state.
        for i in range(num_1f1b_micropairs):
            # Perform forward computation
            output_obj = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
            )

            if gpc.is_last_rank(ParallelMode.PIPELINE):
                output_obj_grad = None
            else:
                output_obj_grad = comm.send_forward_recv_backward(
                    output_obj,
                    backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )

            # Add input_obj and output_obj to end of list.
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            # Pop output_obj and output_obj from the start of the list for
            # the backward pass.
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)

            input_obj_grad = self._backward_step(engine, i, input_obj, output_obj, output_obj_grad)

            if i == (num_1f1b_micropairs - 1):
                input_obj = None
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    comm.send_backward(
                        input_obj_grad,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
            else:
                if gpc.is_first_rank(ParallelMode.PIPELINE):
                    input_obj = None
                else:
                    input_obj = comm.send_backward_recv_forward(
                        input_obj_grad,
                        forward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )

        # Run cooldown backward passes.
        for i in range(num_warmup_microsteps):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                output_obj_grad = comm.recv_backward(
                    backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                output_obj_grad = None

            input_obj_grad = self._backward_step(
                engine, num_1f1b_micropairs + i, input_obj, output_obj, output_obj_grad
            )

            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                comm.send_backward(input_obj_grad, scatter_gather_tensors=self.scatter_gather_tensors)

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        return output, label, accum_loss

    @llm_timeout(func_name="nointerleaved_forward_backward_step")
    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Runs non-interleaved 1F1B schedule, with communication between pipeline stages.
        Returns a tuple with losses if the last stage, an empty tuple otherwise.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                Whether run forward step only. Default is false. If true, no backward will be run.
            return_loss (bool, optional): Whether returns the loss value. Default is true.
            return_output_label (bool, optional): If False, the output and label won't be returned.
        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """

        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        # Load data first
        self.load_batch(engine, data_iter)

        if forward_only:
            return self._forward_only_step(engine, return_loss, return_output_label)
        else:
            return self._forward_backward_step(engine, return_loss, return_output_label)


class InterleavedPipelineScheduler(PipelineScheduler):
    """
    Interleaved Pipeline Scheduler.
    """

    def __init__(
        self,
        num_microbatches: int,
        num_chunks: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        communication_overlap: bool = False,
    ):
        """A helper schedule class for pipeline parallelism running environment.
        It uses interleaved 1F1B strategy. Other properties are similar as
        :class:`NonPipelineSchedule`.

        Args:
            num_microbatches (int): The number of microbatches.
            num_chunks (int): The number of model chunks.
            dtype (torch.dtype, optional): The data type of the tensors. Default is torch.float.
            data_process_func (Callable, optional):
                The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
            tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
            scatter_gather_tensors (bool, optional):
                If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
            scheduler_hooks (List[SchedulerHook], optional): List of scheduler hooks. Default is None.
            communication_overlap (bool, optional): Whether to enable communication overlap. Default is False.
        """
        assert (
            num_microbatches % gpc.get_world_size(ParallelMode.PIPELINE) == 0
        ), "num_microbatches must be an integer multiple of pipeline parallel world size"

        assert (
            isinstance(num_chunks, int) and num_chunks > 0
        ), f"expected num_chunks to be an integer and larger than 0, but got {num_chunks}"

        super().__init__(
            num_microbatches,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )

        gpc.set_virtual_pipeline_parallel_size(num_chunks)
        gpc.set_virtual_pipeline_parallel_rank(0)

        self._num_chunks = num_chunks
        self._communication_overlap = communication_overlap
        # switch 1f1b loop runner function according to communication overlap
        self._run_1f1b_loop = (
            self._run_1f1b_loop_with_overlap if communication_overlap else self._run_1f1b_loop_without_overlap
        )

        # states
        self._pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
        self._pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

        self._accum_loss = None
        self._return_tensors = None
        self._input_objs = [[] for _ in range(num_chunks)]
        self._output_objs = [[] for _ in range(num_chunks)]
        self._output_obj_grads = [[] for _ in range(num_chunks)]

        self._input_obj_shapes = [self.tensor_shape for _ in range(num_chunks)]
        self._output_obj_shapes = [None for _ in range(num_chunks)]
        self._send_tensor_shape_flags = [self.tensor_shape is None for _ in range(num_chunks)]

    @property
    def tensor_shape(self) -> torch.Size:
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, tensor_shape: torch.Size):
        self._tensor_shape = tensor_shape
        self._input_obj_shapes = [self._tensor_shape for _ in range(self._num_chunks)]
        self._send_tensor_shape_flags = [self._tensor_shape is None for _ in range(self._num_chunks)]

    def _clear_state(self) -> None:
        self._accum_loss = None
        self._return_tensors = None
        self._input_objs = [[] for _ in range(self._num_chunks)]
        self._output_objs = [[] for _ in range(self._num_chunks)]
        self._output_obj_grads = [[] for _ in range(self._num_chunks)]

        self._input_obj_shapes = [self.tensor_shape for _ in range(self._num_chunks)]
        self._output_obj_shapes = [None for _ in range(self._num_chunks)]
        self._send_tensor_shape_flags = [self.tensor_shape is None for _ in range(self._num_chunks)]

    def load_batch(self, engine, data_iter):
        super().load_batch(engine, data_iter)
        # overwrite microbatch_offset, since model chunks load the same microbatch, and should tract the offset
        self.microbatch_offset = [0 for _ in range(self._num_chunks)]

    def load_micro_batch(self, model_chunk_id):
        micro_batch_data, micro_batch_label = self._load_micro_batch(
            data=self.batch_data,
            label=self.batch_label,
            offset=self.microbatch_offset[model_chunk_id],
            micro_bsz=self.microbatch_size,
        )
        micro_batch_data["label"] = micro_batch_label
        self.microbatch_offset[model_chunk_id] += self.microbatch_size
        return move_to_device(micro_batch_data)

    def _forward_step(self, engine, chunk_id):
        """Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current
                pipeline stage.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        if gpc.is_pipeline_first_stage() and len(self._input_objs[chunk_id]) == len(self._output_objs[chunk_id]):
            self._input_objs[chunk_id].append(None)
        input_obj = self._input_objs[chunk_id][-1]

        micro_batch_data = self.load_micro_batch(chunk_id)
        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)

        self._call_hooks("before_forward", data)
        output_obj = self._call_engine(engine.model[chunk_id], data)
        # Convert output_obj to fp32 when last model chunk of last stage
        if gpc.is_pipeline_last_stage(ignore_virtual=False) and isinstance(engine.model[chunk_id], NaiveAMPModel):
            output_obj = engine.model[chunk_id].convert_to_fp32(output_obj)
        self._call_hooks("after_forward", output_obj)

        if gpc.is_pipeline_last_stage():
            self._call_hooks("post_helper_func", output_obj, label)

            if self._return_tensors is not None:
                self._return_tensors.append((output_obj, label))
            if self._accum_loss is not None:
                self._call_hooks("before_criterion", output_obj, label)
                loss = self._call_engine_criterion(engine, output_obj, label)
                self._call_hooks("after_criterion", loss)

                loss_reduced = loss / self.num_microbatches
                self._accum_loss.add_(loss_reduced.detach())
                output_obj = loss_reduced

        self._output_objs[chunk_id].append(output_obj)

        return output_obj

    def _backward_step(self, engine, chunk_id, step_id):
        """
        Backward step for passed-in model. If it is the last stage, the input tensor
        is obtained from the previous forward step, otherwise the passed-in input_obj is used.
        Returns input tensor gradient. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
            step_id (int): The current step id.

        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: input tensor gradient.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        if gpc.is_pipeline_last_stage() and len(self._output_obj_grads[chunk_id]) == 0:
            self._output_obj_grads[chunk_id].append(None)

        input_obj = self._input_objs[chunk_id].pop(0)
        output_obj = self._output_objs[chunk_id].pop(0)
        output_obj_grad = self._output_obj_grads[chunk_id].pop(0)

        input_obj_grad = super()._backward_step(engine, step_id, input_obj, output_obj, output_obj_grad)

        return input_obj_grad

    def _get_chunk_by_microbatch(self, step_id: int, backward: bool = False) -> int:
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = step_id % (self._pp_size * self._num_chunks)
        chunk_id = microbatch_id_in_group // self._pp_size

        if backward:
            chunk_id = self._num_chunks - chunk_id - 1

        return chunk_id

    def _get_current_microbatch_id(self, step_id: int) -> int:
        # format:
        # microstep_id : 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        # microbatch_id: 1  2  3  4  1  2  3  4  5  6  7  8  5  6  7  8
        num_microbatch_group = step_id // (self._pp_size * self._num_chunks)
        step_id_in_group = step_id % (self._pp_size * self._num_chunks)

        microbatch_id = num_microbatch_group * self._pp_size + step_id_in_group % self._pp_size

        return microbatch_id

    def _run_warmup_loop(
        self,
        engine: Engine,
        num_microsteps: int,
        num_warmup_microsteps: int,
        receive_extra_backward: bool = False,
        forward_only: bool = False,
    ) -> None:
        """
        Run the warm-up loop and prepare data for the 1F1B stage.

        During the warm-up process, for each execution, it first performs a forward computation,
        and then sends the computation result to the next stage.
        It also receives data for the next forward computation.
        Since the input for the first forward computation is not considered initially,
        it needs to receive data once at the beginning.

        After the warm-up is completed, we need to prepare data for the 1F1B stage.
        The data preparation process should be consistent with the communication method of the 1F1B stage.

        Args:
            engine (Engine): The engine to run the warm-up loop.
            num_microsteps (int): The total number of microsteps.
            num_warmup_microsteps (int): The number of warm-up microsteps.
            receive_extra_backward (bool, optional): Whether to receive extra backward input for the 1F1B stage.
                                                     Default is False.
            forward_only (bool, optional): Whether to only perform forward pass. Default is False.
        """
        if not gpc.is_pipeline_first_stage():
            if self._input_obj_shapes[0] is None:
                self._input_obj_shapes[0] = comm.recv_obj_meta(self._input_obj_shapes[0])
            self._input_objs[0].append(
                comm.recv_forward(
                    self._input_obj_shapes[0],
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            )
        else:
            self._input_objs[0].append(None)

        for k in range(num_warmup_microsteps):
            chunk_id = self._get_chunk_by_microbatch(k)

            output_obj = self._forward_step(engine, chunk_id)

            if forward_only:
                # when forward-only, no need to save tensors for a backward pass
                self._input_objs[chunk_id].pop()
                self._output_objs[chunk_id].pop()

            if not gpc.is_pipeline_last_stage():
                if isinstance(output_obj, torch.Tensor):
                    self._output_obj_shapes[chunk_id] = output_obj.shape
                else:
                    self._output_obj_shapes[chunk_id] = [out_tensor.shape for out_tensor in output_obj]

                if self._send_tensor_shape_flags[chunk_id]:
                    comm.send_obj_meta(output_obj)
                    self._send_tensor_shape_flags[chunk_id] = False  # send only once for each chunk.

            # Determine if tensor should be received from previous stage.
            next_forward_chunk_id = self._get_chunk_by_microbatch(k + 1)

            with switch_virtual_pipeline_parallel_rank(next_forward_chunk_id):
                if not gpc.is_pipeline_first_stage() and self._input_obj_shapes[next_forward_chunk_id] is None:
                    self._input_obj_shapes[next_forward_chunk_id] = comm.recv_obj_meta()
                if k == (num_microsteps - 1) or gpc.is_pipeline_first_stage():
                    input_shape = None
                else:
                    input_shape = self._input_obj_shapes[next_forward_chunk_id]

            # Don't send tensor downstream if on last stage.
            if gpc.is_pipeline_last_stage():
                output_obj = None

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if k != (num_warmup_microsteps - 1) or not receive_extra_backward:
                # Normal warm-up communication process, or no need to prepare backward input for the 1F1B stage
                input_obj = comm.send_forward_recv_forward(
                    output_obj,
                    input_shape,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                # Receive output_obj_grad for next backward, if receive_extra_backward is True.
                if self._communication_overlap:
                    # In this case, we should handle forward and backward communication separately, consistent with the
                    # overlap version of the 1F1B stage
                    input_obj = comm.send_forward_recv_forward(
                        output_obj,
                        input_shape,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    output_obj_grad = comm.send_backward_recv_backward(
                        None,  # nothing to send
                        self._output_obj_shapes[self._num_chunks - 1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    self._output_obj_grads[self._num_chunks - 1].append(output_obj_grad)
                else:
                    # In this case, we should handle forward and backward communication together, consistent with the
                    # non-overlap version of the 1F1B stage
                    input_obj, output_obj_grad = comm.send_forward_backward_recv_forward_backward(
                        output_obj,
                        None,  # no backward grad to send
                        input_shape,
                        self._output_obj_shapes[self._num_chunks - 1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    self._output_obj_grads[self._num_chunks - 1].append(output_obj_grad)

            self._input_objs[next_forward_chunk_id].append(input_obj)

    def _run_1f1b_loop_with_overlap(
        self,
        engine: Engine,
        num_warmup_microsteps: int,
        num_1f1b_micropairs: int,
        all_warmup_microsteps: bool = False,
    ) -> None:
        """
        Run the 1F1B loop with overlap.

        The 1F1B loop with overlap consists of the following steps:
        1. Perform the forward pass.
        2. Check if the backward input is ready.
        3. Send the forward output and receive the forward input for the next iteration.
        4. Perform the backward pass.
        5. Check if the forward input is ready.
        6. Send the backward output and receive the backward input for the next iteration.

        Args:
            engine (Engine): The engine to run the 1F1B loop.
            num_warmup_microsteps (int): The number of warm-up microsteps.
            num_1f1b_micropairs (int): The number of 1F1B micropairs.
            all_warmup_microsteps (bool, optional): Whether to run all warm-up microsteps. Default is False.
        """

        backward_async_communicator = None

        # Run 1F1B in steady state.
        for k in range(num_1f1b_micropairs):
            forward_microstep_id = k + num_warmup_microsteps
            backward_microstep_id = k
            forward_chunk_id = self._get_chunk_by_microbatch(forward_microstep_id)
            backward_chunk_id = self._get_chunk_by_microbatch(backward_microstep_id, backward=True)

            # 1. Forward pass.
            output_obj = self._forward_step(engine, forward_chunk_id)

            # 2. Check if the backward input is ready.
            if backward_async_communicator is not None:
                output_obj_grad = backward_async_communicator.wait_and_receive()

                if backward_async_communicator.need_receive:
                    self._output_obj_grads[backward_chunk_id].append(output_obj_grad)

            # 3. Send the forward outputs and receive the forward inputs from the previous rank.

            # Check if it is the last model chunk of the last pipeline stage, no need to send forward output.
            gpc.set_virtual_pipeline_parallel_rank(forward_chunk_id)
            if gpc.is_pipeline_last_stage():
                output_obj = None

            # Check if it needs to receive the results from the previous rank.
            next_forward_chunk_id = self._get_chunk_by_microbatch(forward_microstep_id + 1)
            with switch_virtual_pipeline_parallel_rank(next_forward_chunk_id):
                if gpc.is_pipeline_first_stage() or k == num_1f1b_micropairs - 1:
                    input_obj_shape = None
                else:
                    input_obj_shape = self._input_obj_shapes[next_forward_chunk_id]

            forward_async_communicator = comm.AsynCommunicator(
                output_obj,
                input_obj_shape,
                self.dtype,
                self.scatter_gather_tensors,
                forward=True,
            )
            forward_async_communicator.start()

            # 5. Backward pass.

            input_obj_grad = self._backward_step(engine, backward_chunk_id, backward_microstep_id)

            input_obj = forward_async_communicator.wait_and_receive()
            if forward_async_communicator.need_receive:
                self._input_objs[next_forward_chunk_id].append(input_obj)

            # 6. Send the backward output and receive the backward input for the next iteration.
            gpc.set_virtual_pipeline_parallel_rank(backward_chunk_id)
            if gpc.is_pipeline_first_stage():
                input_obj_grad = None

            next_backward_chunk_id = self._get_chunk_by_microbatch(backward_microstep_id + 1, backward=True)
            with switch_virtual_pipeline_parallel_rank(next_backward_chunk_id):
                if gpc.is_pipeline_last_stage():
                    output_obj_shape = None
                else:
                    output_obj_shape = self._output_obj_shapes[next_backward_chunk_id]

            backward_async_communicator = comm.AsynCommunicator(
                input_obj_grad,
                output_obj_shape,
                self.dtype,
                self.scatter_gather_tensors,
                forward=False,
            )
            backward_async_communicator.start()

        if all_warmup_microsteps:
            if not gpc.is_pipeline_last_stage():
                self._output_obj_grads[self._num_chunks - 1].append(
                    comm.recv_backward(
                        self._output_obj_shapes[self._num_chunks - 1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                )
            else:
                self._output_obj_grads[self._num_chunks - 1].append(None)
        else:
            output_obj_grad = backward_async_communicator.wait_and_receive()
            if backward_async_communicator.need_receive:
                backward_chunk_id = self._get_chunk_by_microbatch(num_1f1b_micropairs, backward=True)
                self._output_obj_grads[backward_chunk_id].append(output_obj_grad)

    def _run_1f1b_loop_without_overlap(
        self,
        engine: Engine,
        num_warmup_microsteps: int,
        num_1f1b_micropairs: int,
        all_warmup_microsteps: bool = False,
    ) -> None:
        """
        Run the 1F1B loop without overlap.

        The 1F1B loop without overlap consists of the following steps:
        1. Perform the forward pass.
        2. Perform the backward pass.
        3. Send the forward output of this iteration to the next stage, and send the backward output of this iteration
           to the previous stage, and receive the forward and backward inputs for the next iteration.

        Args:
            engine (Engine): The engine to use for computation.
            num_warmup_microsteps (int): The number of warmup microsteps.
            num_1f1b_micropairs (int): The number of 1F1B micro-pairs.
            all_warmup_microsteps (bool, optional): Whether to run all warmup microsteps. Defaults to False.
        """
        for k in range(num_1f1b_micropairs):
            # Forward pass.
            forward_microstep_id = k + num_warmup_microsteps
            forward_chunk_id = self._get_chunk_by_microbatch(forward_microstep_id)
            output_obj = self._forward_step(engine, forward_chunk_id)

            # Backward pass.
            backward_microstep_id = k
            backward_chunk_id = self._get_chunk_by_microbatch(backward_microstep_id, backward=True)
            input_obj_grad = self._backward_step(engine, backward_chunk_id, backward_microstep_id)

            # Send output_obj and input_obj_grad, receive input_obj
            # and output_obj_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set obj to None.
            gpc.set_virtual_pipeline_parallel_rank(forward_chunk_id)
            if gpc.is_pipeline_last_stage():
                output_obj = None

            gpc.set_virtual_pipeline_parallel_rank(backward_chunk_id)
            if gpc.is_pipeline_first_stage():
                input_obj_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            next_forward_chunk_id = self._get_chunk_by_microbatch(forward_microstep_id + 1)
            with switch_virtual_pipeline_parallel_rank(next_forward_chunk_id):
                if gpc.is_pipeline_first_stage() or k == num_1f1b_micropairs - 1:
                    recv_prev = False
                else:
                    recv_prev = True

            next_backward_chunk_id = self._get_chunk_by_microbatch(backward_microstep_id + 1, backward=True)
            with switch_virtual_pipeline_parallel_rank(next_backward_chunk_id):
                if gpc.is_pipeline_last_stage():
                    recv_next = False
                else:
                    recv_next = True

            input_shape = self._input_obj_shapes[next_forward_chunk_id] if recv_prev else None
            output_shape = self._output_obj_shapes[next_backward_chunk_id] if recv_next else None

            # Communicate objs.
            input_obj, output_obj_grad = comm.send_forward_backward_recv_forward_backward(
                output_obj,
                input_obj_grad,
                input_shape,
                output_shape,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )

            # Put input_obj and output_obj_grad in data structures in the
            # right location.
            if recv_prev:
                self._input_objs[next_forward_chunk_id].append(input_obj)
            if recv_next:
                self._output_obj_grads[next_backward_chunk_id].append(output_obj_grad)

        # receive necessary data for next cooldown loop
        if all_warmup_microsteps:
            if not gpc.is_pipeline_last_stage():
                self._output_obj_grads[self._num_chunks - 1].append(
                    comm.recv_backward(
                        self._output_obj_shapes[self._num_chunks - 1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                )
            else:
                self._output_obj_grads[self._num_chunks - 1].append(None)

    def _run_cooldown_loop(self, engine: Engine, num_microsteps: int, num_1f1b_micropairs: int) -> None:
        """
        Run the cooldown loop.

        The cooldown loop consists of the following steps:
        1. Perform the backward step.
        2. Send the backward output to the next stage and receive inputs for next backward.

        Args:
            engine (Engine): The engine to use for computation.
            num_microsteps (int): The total number of microsteps.
            num_1f1b_micropairs (int): The number of 1F1B micro-pairs.
        """
        for k in range(num_1f1b_micropairs, num_microsteps):
            chunk_id = self._get_chunk_by_microbatch(k, backward=True)

            input_obj_grad = self._backward_step(engine, chunk_id, k)

            next_backward_chunk_id = self._get_chunk_by_microbatch(k + 1, backward=True)

            if k != (num_microsteps - 1) and not (
                gpc.is_pipeline_last_stage(ignore_virtual=True) and next_backward_chunk_id == (self._num_chunks - 1)
            ):
                output_shape = self._output_obj_shapes[next_backward_chunk_id]
            else:
                output_shape = None

            self._output_obj_grads[next_backward_chunk_id].append(
                comm.send_backward_recv_backward(
                    input_obj_grad,
                    output_shape,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            )

    def _forward_only_step(self, engine: Engine):
        num_microsteps = self.num_microbatches * self._num_chunks
        num_warmup_microsteps = num_microsteps

        self._run_warmup_loop(
            engine,
            num_microsteps,
            num_warmup_microsteps,
            receive_extra_backward=False,
            forward_only=True,
        )

    def _forward_backward_step(self, engine: Engine):
        # Compute number of warmup and remaining microbatches.
        all_warmup_microsteps = False
        num_microsteps = self.num_microbatches * self._num_chunks

        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if self.num_microbatches == self._pp_size:
            num_warmup_steps = num_microsteps
            all_warmup_microsteps = True
        else:
            num_warmup_steps = (self._pp_size - self._pp_rank - 1) * 2
            num_warmup_steps += (self._num_chunks - 1) * self._pp_size
            num_warmup_steps = min(num_warmup_steps, num_microsteps)
        num_1f1b_micropairs = num_microsteps - num_warmup_steps

        # We usually need to prepare an extra backward data for the 1F1B stage when the WarmUp stage ends,
        # because the 1F1B stage typically performs one forward and backward pass together,
        # except in the following cases:
        receive_extra_backward = not (
            all_warmup_microsteps  # Only warmup microsteps
            or gpc.is_pipeline_last_stage(ignore_virtual=True)  # The rank is the last pipeline stage
        )

        # 1. Warmup
        self._run_warmup_loop(
            engine,
            num_microsteps,
            num_warmup_steps,
            receive_extra_backward=receive_extra_backward,
        )

        # 2. 1F1B
        self._run_1f1b_loop(
            engine,
            num_warmup_steps,
            num_1f1b_micropairs=num_1f1b_micropairs,
            all_warmup_microsteps=all_warmup_microsteps,
        )

        # 3. Cooldown
        self._run_cooldown_loop(engine, num_microsteps, num_1f1b_micropairs=num_1f1b_micropairs)

    @llm_timeout(func_name="interleaved_forward_backward_step")
    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Run interleaved 1F1B schedule (model split into model chunks), with
        communication between pipeline stages as needed.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                Whether run forward step only. Default is false. If true, no backward will be run.
            return_loss (bool, optional): Whether returns the loss value. Default is true.
            return_output_label (bool, optional): If False, the output and label won't be returned.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
                The loss would be returned only in the last stage.
        """
        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        gpc.set_virtual_pipeline_parallel_rank(0)

        self.load_batch(engine, data_iter)

        if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True):
            self._accum_loss = torch.zeros(1, device=get_current_device())
        if return_output_label:
            self._return_tensors = []

        if forward_only:
            self._forward_only_step(engine)
        else:
            self._forward_backward_step(engine)

        if return_output_label and len(self._return_tensors) > 0:
            output, label = pack_return_tensors(self._return_tensors)
        else:
            output, label = (None, None)
        accum_loss = self._accum_loss

        self._clear_state()

        return output, label, accum_loss
