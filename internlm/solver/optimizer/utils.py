#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.utils.common import get_tensor_norm, move_norm_to_cuda
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_model_parallel_parameter

logger = get_logger(__file__)

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    APEX_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    logger.warning("The torch implementation for cal_l2norm is slower than apex. Please note this!")
    APEX_AVAILABLE = False

inf = math.inf


def flatten(input_):
    return _flatten_dense_tensors(input_)


def unflatten(flat, tensors):
    return _unflatten_dense_tensors(flat, tensors)


def get_grad_accumulate_object(tensor):
    """
    Return the AccumulateGrad of the input tensor
    """

    # grad_fn reference:
    # https://discuss.pytorch.org/t/in-the-grad-fn-i-find-a-next-functions-but-i-dont-understand-the-meaning-of-the-attribute/24463
    # expand_as reference: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
    #
    # `next_functions` will return the backward graph where
    # the first element is the AccumulateGrad of the leaf nodes.
    # we want to get the AccumulateGrad of the input tensor instead of the leaf
    # node in the whole computation graph.
    # Therefore, we call expand_as to create a dummy graph
    # where tensor_tmp and tensor indeed point to the same object.
    # You can check this by print(tensor.data_ptr() == tensor_tmp.data_ptr())
    tensor_tmp = tensor.expand_as(tensor)
    grad_acc_obj = tensor_tmp.grad_fn.next_functions[0][0]
    return grad_acc_obj


def split_half_float_double(tensor_list):
    dtypes = ["torch.cuda.HalfTensor", "torch.cuda.FloatTensor", "torch.cuda.DoubleTensor", "torch.cuda.BFloat16Tensor"]
    buckets = []
    for _, dtype in enumerate(dtypes):
        bucket = [t for t in tensor_list if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def reduce_tensor(tensor, dtype=None, dst_rank=None, parallel_mode=ParallelMode.DATA):
    """
    Reduce the tensor in the data parallel process group

    :param tensor: A tensor object to reduce/all-reduce
    :param dtype: The data type used in communication
    :param dst_rank: The source rank for reduce. If dst_rank is None,
    :param parallel_mode: Communication parallel mode
    all-reduce will be used instead of reduce. Default is None.

    :type tensor: torch.Tensor
    :type dtype: torch.dtype, optional
    :type dst_rank: int, optional
    :type parallel_mode: ParallelMode, optional
    """
    # use the original dtype
    if dtype is None:
        dtype = tensor.dtype

    # cast the data to specified dtype for reduce/all-reduce
    if tensor.dtype != dtype:
        tensor_to_reduce = tensor.to(dtype)
    else:
        tensor_to_reduce = tensor

    world_size = gpc.get_world_size(parallel_mode)
    group = gpc.get_group(parallel_mode)
    tensor_to_reduce.div_(world_size)

    # if rank is None, all reduce will be used
    # else, reduce is used
    use_all_reduce = dst_rank is None

    if use_all_reduce:
        dist.all_reduce(tensor_to_reduce, group=group)
    else:
        ranks_in_group = gpc.get_ranks_in_group(parallel_mode)
        global_rank = ranks_in_group[dst_rank]
        dist.reduce(tensor=tensor_to_reduce, dst=global_rank, group=group)

    # recover the original dtype
    if tensor.dtype != dtype and tensor is not tensor_to_reduce:
        local_rank = gpc.get_local_rank(parallel_mode)
        if use_all_reduce or dst_rank == local_rank:
            tensor.copy_(tensor_to_reduce)

    return tensor


def has_inf_or_nan(tensor):
    try:
        # if tensor is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as tensor
        # (which is true for some recent version of pytorch).
        tensor_sum = float(tensor.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # tensor_sum = float(tensor.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if tensor_sum == float("inf") or tensor_sum == -float("inf"):
            return True
        return False


def release_param_grad(tensor_list):
    for tensor in tensor_list:
        tensor.grad = None


def sync_param(flat_tensor, tensor_list):
    """
    Synchronize the flattened tensor and unflattened tensor list. When
    a list of tensor are flattened with `torch._utils._unflatten_dense_tensors`,
    a new tensor is created. Thus, the flat tensor and original tensor list do not
    share the same memory space. This function will update the tensor list so that
    they point to the same value.

    :param flat_tensor: A flat tensor obtained by calling `torch._utils._unflatten_dense_tensors` on a tensor lsit
    :param tensor_list: A list of tensors corresponding to the flattened tensor
    :type flat_tensor: torch.Tensor
    :type tensor_list: List[torch.Tensor]
    """
    updated_params = unflatten(flat_tensor, tensor_list)

    # update the tensor data
    for p, q in zip(tensor_list, updated_params):
        p.data = q.data


def multi_tensor_l2norm_torch(tensor_list, per_tensor):
    # Convert tensor_list elements to torch.float32
    tensor_list = [tensor.float() for tensor in tensor_list]
    norms_tensor = torch.stack([torch.norm(tensor, p=2) for tensor in tensor_list])
    l2_norm = torch.norm(norms_tensor, p=2).unsqueeze(0)

    if per_tensor:
        per_tensor_norm = norms_tensor
    else:
        per_tensor_norm = torch.Tensor([]).to(norms_tensor.device)

    return l2_norm, per_tensor_norm


def calc_l2_norm(grads):
    norm = 0.0
    if len(grads) > 0:
        if APEX_AVAILABLE:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            norm, _ = multi_tensor_applier(
                amp_C.multi_tensor_l2norm, dummy_overflow_buf, [grads], False  # no per-parameter norm
            )
        else:
            norm, _ = multi_tensor_l2norm_torch(grads, False)
    return norm


def calc_lp(grads, norm_type):
    norm = 0.0
    for grad in grads:
        grad_norm = torch.norm(grad, norm_type)
        norm += grad_norm**norm_type
    return norm


def compute_norm(gradients, parameters, last_stage=False, previous_norm=None, norm_type=2):
    """Get the norm
    Arguments:
        gradients (Iterable[Tensor]): The gradient value.
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters, need total_norm**(1/norm) before using.
    """

    enable_cuda_kernels = gradients[0].device.type == "cuda"
    # Norm parameters.
    norm_type = float(norm_type)

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_cuda = torch.FloatTensor([float(total_norm)], device=gradients[0].device)

        if last_stage is False:
            return total_norm_cuda

        if previous_norm is not None:
            total_norm_cuda = max(total_norm_cuda, previous_norm)

        # Take max across all model-parallel GPUs.
        if gpc.get_world_size(ParallelMode.MODEL) > 1:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MODEL))
        total_norm = total_norm_cuda[0].item()
    else:
        tensor_parallel_grads = []
        for g, p in zip(gradients, parameters):
            # TODO: consider the pipeline shared parameter
            if (
                gpc.is_initialized(ParallelMode.PIPELINE)
                and hasattr(p, "pipeline_shared_module_pg")
                and dist.get_rank(p.pipeline_shared_module_pg) == 0
            ):  # if shared between different pipe, only count o
                tensor_parallel_grads.append(g.data.float())
            elif (
                gpc.is_initialized(ParallelMode.PIPELINE)
                and hasattr(p, "pipeline_shared_module_pg")
                and dist.get_rank(p.pipeline_shared_module_pg) != 0
            ):
                continue
            elif (
                gpc.is_initialized(ParallelMode.TENSOR)
                and not is_model_parallel_parameter(p)
                and gpc.get_local_rank(ParallelMode.TENSOR) == 0
            ):  # if not used in each chunk, such as layernorm
                tensor_parallel_grads.append(g.data.float())
            elif is_model_parallel_parameter(p):
                tensor_parallel_grads.append(g.data.float())
            elif gpc.get_local_rank(ParallelMode.TENSOR) != 0:
                continue
            else:
                raise RuntimeError("Should not arrive here")

        if norm_type == 2.0 and enable_cuda_kernels:
            tensor_parallel_norm = calc_l2_norm(tensor_parallel_grads) ** norm_type
        else:
            tensor_parallel_norm = calc_lp(tensor_parallel_grads, norm_type)

        # If norm is type of float, then we convert them into torch.Tensor.
        tensor_parallel_norm = get_tensor_norm(tensor_parallel_norm, enable_cuda_kernels)
        # If grads are on CPU, the norms is also on CPU. Cast them to CUDA tensors
        if not enable_cuda_kernels:
            tensor_parallel_norm = move_norm_to_cuda(tensor_parallel_norm)

        total_norm = tensor_parallel_norm

        if last_stage is False:
            return total_norm

        if previous_norm is not None:
            total_norm = total_norm + previous_norm

        # Sum across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.MODEL):
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.MODEL))

        # This is because we use zero1, so we need to use this reduction.
        # TODO: Check zero group to be a subset of dp group.
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.ZERO1))

        if torch.is_tensor(total_norm):
            total_norm = total_norm.item()

    # Scale.
    if total_norm == float("inf") or total_norm == -float("inf"):
        total_norm = -1

    return total_norm


class BaseGradScaler(ABC):
    """A base class for the gradient scaler.

    Args:
        initial_scale (float): the initial loss scale
    """

    def __init__(self, initial_scale: float):
        assert initial_scale > 0
        self._scale = torch.cuda.FloatTensor([initial_scale])

    @property
    def scale(self) -> Tensor:
        """Returns the loss scale."""

        return self._scale

    @property
    def inv_scale(self) -> Tensor:
        """Returns the inverse of the loss scale."""

        return self._scale.double().reciprocal().float()

    def state_dict(self) -> Dict:
        """Returns the states of the gradient scaler as a dict object."""

        state_dict = dict()
        state_dict["scale"] = self.scale
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load the states of the gradient scaler from a dict object.

        Args:
            state_dict (dict): the states of the gradient scaler
        """

        self._scale = state_dict["scale"]

    @abstractmethod
    def update(self, overflow: bool) -> None:
        """Update the loss scale.

        Args:
            overflow (bool): whether overflow occurs
        """

        pass


class DynamicGradScaler(BaseGradScaler):
    """A gradient scaler which uses dynamic loss scale

    Args:
        initial_scale (float): the initial loss scale, defaults to 2**16
        growth_factor (float): the multiplication factor for increasing loss scale, defaults to 2
        backoff_factor (float): the multiplication factor for decreasing loss scale, defaults to 0.5
        growth_interval (int): the number of steps to increase loss scale when no overflow occurs, defaults to 1000
        min_scale (float): the minimum loss scale, defaults to None
        max_scale (float): the maximum loss scale, defaults to None
        hysteresis (int):  the number of overflows before decreasing loss scale, defaults to 2
    """

    def __init__(
        self,
        initial_scale: float = 2**16,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        min_scale: Optional[float] = None,
        max_scale: Optional[float] = None,
        hysteresis: int = 2,
    ):
        super().__init__(initial_scale)
        if min_scale:
            self._min_scale = torch.cuda.FloatTensor([min_scale])
        else:
            self._min_scale = None

        if max_scale:
            self._max_scale = torch.cuda.FloatTensor([max_scale])
        else:
            self._max_scale = None

        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_step = 0
        self._hysteresis = hysteresis
        self._hysteresis_step = 0
        self._sanity_checks()

    def _sanity_checks(self) -> None:
        """Check if the arguments are correct."""

        if self._min_scale:
            assert self._min_scale > 0, "The minimum gradient scale cannot be zero or negative"
        if self._max_scale:
            assert self._min_scale > 0, "The maximum gradient scale cannot be zero or negative"
        assert self._growth_factor > 1, "The growth factor cannot be equal or smaller than 1"
        assert self._backoff_factor < 1 and self._backoff_factor > 0, "The backoff factor must be between 0 and 1"
        assert self._hysteresis >= 0, "The hysteresis cannot be negative"

    def update(self, overflow: bool) -> None:
        """Update the loss scale.

        Args:
            overflow (bool): whether overflow occurs
        """
        if overflow:
            self._hysteresis_step += 1
            self._growth_step = 0

            if self._hysteresis_step >= self._hysteresis:
                self._backoff_scale()
                if gpc.is_rank_for_log():
                    logger.warning(f"Overflow occurs, the loss scale is adjusted to {self.scale.item()}")
        else:
            self._growth_step += 1
            if self._growth_step == self._growth_interval:
                self._growth_step = 0
                self._hysteresis_step = 0
                self._grow_scale()
                if gpc.is_rank_for_log():
                    logger.warning(
                        f"No overflow for consecutive {self._growth_interval} steps, "
                        f"the loss scale is adjusted to {self.scale.item()}",
                    )

    def _backoff_scale(self) -> None:
        """Decrease the loss scale"""

        self._scale = self._scale * self._backoff_factor
        if self._min_scale:
            self._scale = torch.max(self._scale, self._min_scale)

    def _grow_scale(self) -> None:
        """Increase the loss scale"""

        self._scale = self._scale * self._growth_factor
        if self._max_scale:
            self._scale = torch.min(self._scale, self._max_scale)

    def state_dict(self):
        """Returns the states of the gradient scaler as a dict object."""

        state_dict = dict()
        state_dict["_scale"] = self._scale.item()
        state_dict["_growth_step"] = self._growth_step
        state_dict["_hysteresis_step"] = self._hysteresis_step

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the states of the gradient scaler from a dict object.

        Args:
            state_dict (dict): the states of the gradient scaler
        """

        self._scale = self._scale.fill_(state_dict["_scale"])
        self._growth_step = state_dict["_growth_step"]
        self._hysteresis_step = state_dict["_hysteresis_step"]


GroupedPartitionParameters = List[Dict[str, List[Tensor]]]


class ModelParatitionHandler(ABC):
    """
    ModelParatitionHandler
    """

    @abstractmethod
    def partition(
        self, group_id: int, world_size: int, param_record_per_rank: List[List[str]]
    ) -> (List[List[Tensor]], Set[int]):
        """partition model parameters for each rank"""

    @abstractmethod
    def resync_flat_parameters(self, group_id: int, rank: int, flat_tensor: Tensor) -> None:
        """resync flat parameter with model parameters"""

    @abstractmethod
    def register_sync_parameters_hook(self, model: nn.Module) -> None:
        """register hooks to sync model parameters"""


class AsyncModelPartitionHandler(ModelParatitionHandler):
    """
    AsyncModelPartitionHandler
    """

    def __init__(
        self,
        partition_scheme: GroupedPartitionParameters,
        process_group_mode: ParallelMode,
    ) -> None:
        num_groups = len(partition_scheme)

        assert num_groups == 1, "Only one parameter group is supported."

        self._num_groups = num_groups
        self._process_group_mode = process_group_mode
        self._paration_rank_map = [{} for _ in range(num_groups)]
        self._grouped_partition_params = partition_scheme
        self._grouped_partition_flat_params = self._flatten_partition_parameters(
            partition_scheme
        )  # (group, part) -> flat_tensor
        self._grouped_rank_flat_params = [None for _ in range(num_groups)]  # (group, rank) -> flat_tensors

        # states
        self.params_synced = True

        self._current_layer_idx = 0
        self._partitions_to_process = list(partition_scheme[0])
        self._async_handle = None

    def _sync_flat_parameters(self, flat_tensor: Tensor, origin_tensors: List[Tensor]) -> None:
        updated_params = torch._utils._unflatten_dense_tensors(flat_tensor, origin_tensors)

        for origin, updated in zip(origin_tensors, updated_params):
            origin.data = updated.data

    def resync_flat_parameters(self, group_id: int, rank: int, flat_tensor: Tensor) -> None:
        rank_flat_params = self._grouped_rank_flat_params[group_id][rank]

        self._sync_flat_parameters(flat_tensor, rank_flat_params)

    def _flatten_partition_parameters(self, grouped_partition_parameters) -> List[Dict[str, Tensor]]:
        result = []

        def _flatten_and_sync_params(params: List[Tensor]) -> Tensor:
            with torch.no_grad():
                flat_tensor = torch._utils._flatten_dense_tensors(params)
            flat_tensor = flat_tensor.data.cuda()

            # sync with origin paramsters
            self._sync_flat_parameters(flat_tensor, params)

            return flat_tensor

        for part_params in grouped_partition_parameters:
            group_result = {}

            for part_name, params in part_params.items():
                if len(params) > 1:
                    flat_params = _flatten_and_sync_params(params)
                else:
                    flat_params = params[0]
                group_result[part_name] = flat_params

            result.append(group_result)

        return result

    def _generate_param_record(self, part_name: str, param_id: int, part_param: Tensor) -> str:
        param_size = part_param.size()
        size_strings = [str(param_size[i]) for i in range(len(param_size))]
        return "_".join([f"{part_name}.{param_id}", *size_strings])

    def partition(self, group_id: int, world_size: int, param_record_per_rank: List[List[str]]):
        params_per_rank = [[] for _ in range(world_size)]
        numel_per_rank = [0 for _ in range(world_size)]

        self._grouped_rank_flat_params[group_id] = [[] for _ in range(world_size)]

        for part_name, part_params in self._grouped_partition_params[group_id].items():
            part_flat_tensor = self._grouped_partition_flat_params[group_id][part_name]

            rank_to_go = numel_per_rank.index(min(numel_per_rank))
            params_per_rank[rank_to_go].extend(part_params)
            numel_per_rank[rank_to_go] += part_flat_tensor.numel()

            # for convert helper script
            for idx, part_param in enumerate(part_params):
                param_record_per_rank[rank_to_go].append(self._generate_param_record(part_name, idx, part_param))

            self._paration_rank_map[group_id][part_name] = rank_to_go
            self._grouped_rank_flat_params[group_id][rank_to_go].append(part_flat_tensor)

        if gpc.is_rank_for_log():
            logger.info("Rank%d: number of elements on ranks is %s", gpc.get_global_rank(), numel_per_rank)

        return params_per_rank, set()

    def register_sync_parameters_hook(self, model: nn.Module) -> None:
        # sync parameters before forward.
        def _pre_forward_hook(model: nn.Module, inputs: Any, is_first: bool = False):
            if self.params_synced or self._current_layer_idx + 1 == len(self._partitions_to_process):
                return

            del model, inputs

            for group_id in range(self._num_groups):
                if is_first:
                    part_name = self._partitions_to_process[self._current_layer_idx]
                    src_rank = gpc.get_ranks_in_group(self._process_group_mode)[
                        self._paration_rank_map[group_id][part_name]
                    ]
                    dist.broadcast(
                        self._grouped_partition_flat_params[0][part_name],
                        src=src_rank,
                        group=gpc.get_group(self._process_group_mode),
                    )
                else:
                    part_name = self._partitions_to_process[self._current_layer_idx + 1]
                    src_rank = gpc.get_ranks_in_group(self._process_group_mode)[
                        self._paration_rank_map[group_id][part_name]
                    ]
                    self._async_handle = dist.broadcast(
                        self._grouped_partition_flat_params[0][part_name],
                        src=src_rank,
                        group=gpc.get_group(self._process_group_mode),
                        async_op=True,
                    )

        # wait for parameter synced after forward.
        def _post_forward_hook(model: nn.Module, args: Any, output: Any):
            if self.params_synced:
                return

            del model, args, output

            if self._async_handle is not None:
                self._async_handle.wait()
                self._async_handle = None

            self._current_layer_idx += 1

        # set parameter syncted flag.
        def _post_all_forward_hook(model: nn.Module, args: Any, output: Any):
            if self.params_synced:
                return

            del model, args, output

            self._current_layer_idx = 0
            self.params_synced = True

        if isinstance(model, NaiveAMPModel):
            model = model.model

        # sync parameter for the first layer
        model.register_forward_pre_hook(partial(_pre_forward_hook, is_first=True))
        # set parameter synced flag.
        model.register_forward_hook(_post_all_forward_hook)

        for part_name in self._partitions_to_process:
            try:
                sub_model = model.get_submodule(part_name)
            except AttributeError:
                logger.warning("skip register model parameter sync hook for missing layer %s", part_name)
                continue

            # sync parameter for the next layer async.
            sub_model.register_forward_pre_hook(partial(_pre_forward_hook, is_first=False))
            # check the async communication ready.
            sub_model.register_forward_hook(_post_forward_hook)


class AsyncMultiChunkParatitionHandler(ModelParatitionHandler):
    """
    AsyncMultiChunkParatitionHandler
    """

    def __init__(
        self, num_chunk: int, partition_schemes: List[GroupedPartitionParameters], process_group_mode: ParallelMode
    ) -> None:
        self._num_chunk = num_chunk
        self._handlers = [
            AsyncModelPartitionHandler(partition_schemes[idx], process_group_mode) for idx in range(num_chunk)
        ]

    @property
    def params_synced(self) -> bool:
        _params_synced = True

        for handler in self._handlers:
            _params_synced = _params_synced and handler.params_synced

        return _params_synced

    @params_synced.setter
    def params_synced(self, _synced) -> None:
        for handler in self._handlers:
            handler.params_synced = _synced

    def _sync_flat_parameters(self, flat_tensor: Tensor, origin_tensors: List[Tensor]) -> None:
        updated_params = torch._utils._unflatten_dense_tensors(flat_tensor, origin_tensors)

        for origin, updated in zip(origin_tensors, updated_params):
            origin.data = updated.data

    def resync_flat_parameters(self, group_id: int, rank: int, flat_tensor: Tensor) -> None:
        rank_flat_params = []

        for handler in self._handlers:
            rank_flat_params.extend(handler._grouped_rank_flat_params[group_id][rank])

        self._sync_flat_parameters(flat_tensor, rank_flat_params)

    def partition(self, group_id: int, world_size: int, param_record_per_rank: List[List[str]]):
        params_per_rank = [[] for _ in range(world_size)]
        no_param_ranks = set()

        for handler in self._handlers:
            _params_per_rank, _no_param_ranks = handler.partition(group_id, world_size, param_record_per_rank)
            # merge params_per_rank
            for idx, params in enumerate(params_per_rank):
                params.extend(_params_per_rank[idx])
            # merge no_param_ranks
            no_param_ranks = no_param_ranks & _no_param_ranks

        return params_per_rank, no_param_ranks

    def register_sync_parameters_hook(self, model: nn.Module):
        assert isinstance(model, nn.ModuleList), "model must be a MouduleList"
        assert len(model) == self._num_chunk, f"length of model list({len(model)}) != self._num_chunk{self._num_chunk}"

        for idx, handler in enumerate(self._handlers):
            handler.register_sync_parameters_hook(model[idx])
