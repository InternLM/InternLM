#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.utils.common import move_norm_to_cuda
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
global_layer_norms = {"unknown": 0.0, "embedding": 0.0, "norm": 0.0, "head": 0.0}


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
    dtype_buckets = {
        "torch.cuda.HalfTensor": [],
        "torch.cuda.FloatTensor": [],
        "torch.cuda.DoubleTensor": [],
        "torch.cuda.BFloat16Tensor": [],
    }

    for t in tensor_list:
        dtype = t.type()
        if dtype in dtype_buckets:
            dtype_buckets[dtype].append(t)

    buckets = [bucket for bucket in dtype_buckets.values() if bucket]
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
    # if dtype is None:
    assert dtype is None
    dtype = tensor.dtype

    # cast the data to specified dtype for reduce/all-reduce
    # if tensor.dtype != dtype:
    #     tensor_to_reduce = tensor.to(dtype)
    # else:
    #     tensor_to_reduce = tensor

    # world_size = gpc.get_world_size(parallel_mode)
    # tensor.div_(world_size)
    group = gpc.get_group(parallel_mode)

    # if rank is None, all reduce will be used
    # else, reduce is used
    use_all_reduce = dst_rank is None

    if use_all_reduce:
        handle = dist.all_reduce(tensor=tensor, group=group, op=torch.distributed.ReduceOp.AVG, async_op=True)
    else:
        ranks_in_group = gpc.get_ranks_in_group(parallel_mode)
        global_rank = ranks_in_group[dst_rank]
        handle = dist.reduce(
            tensor=tensor, dst=global_rank, group=group, op=torch.distributed.ReduceOp.AVG, async_op=True
        )

    return handle


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
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grads],
                False,  # no per-parameter norm
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


def compute_norm(
    gradients, parameters, last_stage=False, previous_layer_norms=None, norm_type=2, zero_mode=ParallelMode.ZERO1
):
    """Get the norm
    Arguments:
        gradients (Iterable[Tensor]): The gradient value.
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        gradient norm for all layers, need total_norm**(1/norm) before using.
    """

    enable_cuda_kernels = gradients[0].device.type == "cuda"
    # Norm parameters.
    norm_type = float(norm_type)
    total_layer_norms = copy.deepcopy(global_layer_norms)
    layer_grads = {}
    # Calculate norm.
    if norm_type == inf:
        for g, p in zip(gradients, parameters):
            layer_name = p.layer_name if hasattr(p, "layer_name") else "unknown"
            if layer_name not in layer_grads:
                layer_grads[layer_name] = []
            layer_grads[layer_name].append(g)
        for layer_name, grads in layer_grads.items():
            layer_norm = max(g.data.abs().max() for g in grads)
            total_layer_norms[layer_name] = max(total_layer_norms[layer_name], float(layer_norm))

        if last_stage is False:
            return total_layer_norms

        if previous_layer_norms is not None:
            for key, value in previous_layer_norms.items():
                total_layer_norms[key] = max(value, total_layer_norms[key])

        total_layer_norms_values = move_norm_to_cuda(torch.Tensor(list(total_layer_norms.values())))
        total_layer_norms_keys = list(global_layer_norms.keys())

        # Take max across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.MODEL):
            dist.all_reduce(
                total_layer_norms_values,
                op=dist.ReduceOp.MAX,
                group=gpc.get_group(ParallelMode.MODEL),
            )
        for idx in range(len(total_layer_norms_keys)):
            layer_norm = total_layer_norms_values[idx]
            if torch.is_tensor(layer_norm):
                layer_norm = layer_norm.item()
            total_layer_norms[total_layer_norms_keys[idx]] = layer_norm

    else:
        for g, p in zip(gradients, parameters):
            # TODO: consider the pipeline shared parameter
            layer_name = p.layer_name if hasattr(p, "layer_name") else "unknown"
            if layer_name not in layer_grads:
                layer_grads[layer_name] = []
            if (
                gpc.is_initialized(ParallelMode.PIPELINE)
                and hasattr(p, "pipeline_shared_module_pg")
                and dist.get_rank(p.pipeline_shared_module_pg) == 0
            ):  # if shared between different pipe, only count o
                layer_grads[layer_name].append(g.data.float())
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
                layer_grads[layer_name].append(g.data.float())
            elif is_model_parallel_parameter(p):
                layer_grads[layer_name].append(g.data.float())
            elif gpc.get_local_rank(ParallelMode.TENSOR) != 0:
                continue
            else:
                raise RuntimeError("Should not arrive here")

        # calculate lay norm
        for layer_name, grads in layer_grads.items():
            if norm_type == 2.0 and enable_cuda_kernels:
                layer_norm = calc_l2_norm(grads) ** norm_type
            else:
                layer_norm = calc_lp(grads, norm_type)
            total_layer_norms[layer_name] += layer_norm.item() if torch.is_tensor(layer_norm) else layer_norm

        if last_stage is False:
            return total_layer_norms

        if previous_layer_norms is not None:
            for key, value in previous_layer_norms.items():
                total_layer_norms[key] += value

        # sync layer norm
        # Sum across all model-parallel GPUs.
        total_layer_norms_values = move_norm_to_cuda(torch.Tensor(list(total_layer_norms.values())))
        total_layer_norms_keys = list(total_layer_norms.keys())

        if gpc.is_initialized(ParallelMode.MODEL):
            dist.all_reduce(total_layer_norms_values, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.MODEL))
        dist.all_reduce(total_layer_norms_values, op=dist.ReduceOp.SUM, group=gpc.get_group(zero_mode))

        for idx, layer_name in enumerate(total_layer_norms.keys()):
            layer_norm = total_layer_norms_values[idx]
            if torch.is_tensor(layer_norm):
                layer_norm = layer_norm.item()
            if layer_norm == float("inf") or layer_norm == -float("inf"):
                layer_norm = -1

            if math.isnan(layer_norm):
                layer_norm = -2
            total_layer_norms[layer_name] = layer_norm

    return total_layer_norms


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


class ParamBcastSyncHandler:
    """
    Model Partition Handler for overlap broadcast with forward
    """

    def __init__(self, model: Union[nn.Module, nn.ModuleList]) -> None:
        self._block_to_param = OrderedDict()  # <key: nn.Module> <value: list(param)>
        self._param_to_rank = dict()  # <key: param> <value: rank)>
        self._block_to_rank = dict()  # <key: nn.Module> <value: rank)>
        self._bcast_handles = dict()  # <key: rank> <value: list(bcast handles))>

        zero1_size = gpc.get_world_size(ParallelMode.ZERO1)
        total_param_num = sum(p.numel() for p in model.parameters())
        avg_param_num = total_param_num * 1.0 // zero1_size

        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]

        # record the parameters to transformer/embeding/head/norm block
        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model
            # if gpc.is_rank_for_log():
            # logger.info(_chunk)
            # [ name for name , _ in  model.model.named_children()]
            for name, children in _chunk.named_children():
                # should be the transformer block definaton in modeling_xxx.py
                if isinstance(children, nn.ModuleList):
                    # record the block that a parameter belongs to
                    for idx, block in enumerate(children):
                        # self._block_to_param[f"{name}.{idx}"] = list(block.parameters())
                        self._block_to_param[block] = list(block.parameters())
                        for parameter in self._block_to_param[block]:
                            layer_name = f"{block.__class__.__name__}.{idx}"
                            # if gpc.is_rank_for_log():
                            # logger.info(layer_name)
                            global_layer_norms[layer_name] = 0.0
                            parameter.__setattr__("layer_name", layer_name)
                else:
                    # record the block that a parameter belongs to
                    # self._block_to_param[name] = list(children.parameters())
                    self._block_to_param[children] = list(children.parameters())
                    for parameter in self._block_to_param[children]:
                        layer_name = f"{children.__class__.__name__}"
                        # if gpc.is_rank_for_log():
                        # logger.info(layer_name)
                        # global_layer_norms[layer_name] = 0.0
                        parameter.__setattr__("layer_name", name)

        alloc_num = 0
        rank_to_go = 0

        # process the parameters in block_to_param sequencially,
        # allocate each parameter to a local rank of ParallelMode.ZERO1,
        # NOTE that we do NOT consider following scenarios:
        # 1) whether a parameter is trainable;
        # 2) paramters maybe in different optimizer group
        for block, params in self._block_to_param.items():
            # allocate a model block to a local rank of ParallelMode.ZERO1
            self._block_to_rank[block] = [rank_to_go]
            for p in params:
                alloc_num = alloc_num + p.numel()
                # in this case, allocate the param to next rank if possible
                if alloc_num > avg_param_num * 1.01 and rank_to_go < zero1_size - 1:
                    rank_to_go = rank_to_go + 1
                    alloc_num = 0
                    self._block_to_rank[block].append(rank_to_go)
                # allocate a parameter to a local rank of ParallelMode.ZERO1
                self._param_to_rank[p] = rank_to_go

        # initialize an empty list for _bcast_handles of each rank
        for rank in range(gpc.get_world_size(ParallelMode.ZERO1)):
            self._bcast_handles[rank] = []

        # register_forward_pre_hook for transformer/embeding/norm/xxx block
        self._register_sync_parameters_hook()

    def _register_sync_parameters_hook(self) -> None:
        def _pre_forward_hook(model: nn.Module, inputs: Any):  # pylint: disable=W0613
            bcast_handles = []
            # gather all required broadcast hanles into a list
            for rank in self._block_to_rank[model]:
                bcast_handles.extend(self._bcast_handles[rank])
                # need to clear _bcast_handles since they would be processed later
                self._bcast_handles[rank] = []
            # wait all required broadcast handles to be completed
            for handle in bcast_handles:
                handle.wait()

        # register_forward_pre_hook for transformer/embeding/norm/xxx block
        for block, _ in self._block_to_rank.items():
            block.register_forward_pre_hook(partial(_pre_forward_hook))

    def get_rank_by_param(self, param) -> int:
        return self._param_to_rank[param]

    def add_bcast_handle(self, rank, handle) -> None:
        self._bcast_handles[rank].append(handle)
