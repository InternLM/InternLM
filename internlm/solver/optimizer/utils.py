#!/usr/bin/env python
# -*- encoding: utf-8 -*-

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
from internlm.utils.common import get_current_device, get_tensor_norm, move_norm_to_cuda
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


def calc_zero_grad(grads):
    zero_count = 0
    grad_size = 0
    for grad in grads:
        zero_count += (grad == 0).sum().item()
        grad_size += grad.numel()
    return torch.tensor([zero_count, grad_size])


def get_norm(grads, norm_type, enable_cuda_kernels):
    if norm_type == inf:
        grad_norm = max(g.data.abs().max() for g in grads)
    elif norm_type == 2.0 and enable_cuda_kernels:
        grad_norm = calc_l2_norm(grads) ** norm_type
    else:
        grad_norm = calc_lp(grads, norm_type)
    return grad_norm


def reduce_grads(gradients, parameters, fine_grained=False, only_output=False):
    parallel_grads = []
    if fine_grained:
        parallel_grads = {}

    def append_grad(g, p):
        if fine_grained:
            param_name = p.param_name if hasattr(p, "param_name") else "unknown-padding"
            if param_name not in parallel_grads:
                parallel_grads[param_name] = []
            parallel_grads[param_name].append(g.data.float())
        elif only_output:
            param_name = p.param_name if hasattr(p, "param_name") else "unknown-padding"
            if (
                gpc.config.model["vocab_size"] == g.shape[0] * gpc.get_world_size(ParallelMode.TENSOR)
                and gpc.config.model["hidden_size"] == g.shape[1]
                and "embedding" not in param_name.lower()
            ):
                parallel_grads.append(g.data.float())
        else:
            parallel_grads.append(g.data.float())

    for g, p in zip(gradients, parameters):
        # TODO: consider the pipeline shared parameter
        if (
            gpc.is_initialized(ParallelMode.PIPELINE)
            and hasattr(p, "pipeline_shared_module_pg")
            and dist.get_rank(p.pipeline_shared_module_pg) == 0
        ):  # if shared between different pipe, only count o
            append_grad(g, p)
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
            append_grad(g, p)
        elif is_model_parallel_parameter(p):
            append_grad(g, p)
        elif gpc.get_local_rank(ParallelMode.TENSOR) != 0:
            continue
        else:
            raise RuntimeError("Should not arrive here")
    return parallel_grads


def compute_norm(
    gradients,
    parameters,
    last_stage=False,
    previous_norm=None,
    norm_type=2,
    zero_mode=ParallelMode.ZERO1,
):
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
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.MAX,
                group=gpc.get_group(ParallelMode.MODEL),
            )
        total_norm = total_norm_cuda[0].item()
    else:
        tensor_parallel_grads = reduce_grads(gradients, parameters)

        tensor_parallel_norm = get_norm(tensor_parallel_grads, norm_type, enable_cuda_kernels)

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
            dist.all_reduce(
                total_norm,
                op=dist.ReduceOp.SUM,
                group=gpc.get_group(ParallelMode.MODEL),
            )

        # This is because we use zero1, so we need to use this reduction.
        # TODO: Check zero group to be a subset of dp group.
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(zero_mode))

        if torch.is_tensor(total_norm):
            total_norm = total_norm.item()

    # Need to allreduce(avg) the norms across different ranks because moe params will not be synced during allreduce
    # model and zero have been reduced!!!
    if zero_mode == ParallelMode.EXPERT_DATA:
        pg = gpc.get_group(ParallelMode.EXPERT)
        scaled_norm = total_norm * 1.0 / float(gpc.get_world_size(ParallelMode.DATA))
        scaled_norm_tensor = torch.tensor(scaled_norm, device=get_current_device(), dtype=torch.float)
        dist.all_reduce(scaled_norm_tensor, group=pg)
        total_norm = scaled_norm_tensor.item()

    # Scale.
    if total_norm == float("inf") or total_norm == -float("inf"):
        total_norm = -1

    if math.isnan(total_norm):
        total_norm = -2

    return total_norm


def compute_vocab_grad_norm(
    gradients,
    parameters,
    last_stage=False,
    previous_vocab_grad_norm=None,
    norm_type=2,
    zero_mode=ParallelMode.ZERO1,
):
    enable_cuda_kernels = gradients[0].device.type == "cuda"
    # Norm parameters.
    norm_type = float(norm_type)
    vocab_size = gpc.config.model["vocab_size"]

    param_grads = reduce_grads(gradients, parameters, only_output=True)

    vocab_grad_norm = torch.zeros((vocab_size,), dtype=torch.float32).to(get_current_device())
    if param_grads:
        for grad in param_grads:
            # get grad norm of each vocab
            vocab_slice_size = grad.shape[0]
            local_tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
            for i in range(vocab_slice_size):
                cur_vocab_grad_norm = get_norm([grad[i, :]], norm_type, enable_cuda_kernels)[0]
                vocab_grad_norm[i + vocab_slice_size * local_tp_rank] += get_tensor_norm(
                    cur_vocab_grad_norm, move_to_cuda=True
                )

    if last_stage is False:
        return vocab_grad_norm

    if previous_vocab_grad_norm is not None:
        vocab_grad_norm = vocab_grad_norm + previous_vocab_grad_norm

    if gpc.is_initialized(ParallelMode.MODEL):
        dist.all_reduce(
            vocab_grad_norm,
            op=dist.ReduceOp.SUM,
            group=gpc.get_group(ParallelMode.MODEL),
        )

    dist.all_reduce(vocab_grad_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(zero_mode))

    if zero_mode == ParallelMode.EXPERT_DATA:
        pg = gpc.get_group(ParallelMode.EXPERT)
        scaled_norm = vocab_grad_norm * 1.0 / float(gpc.get_world_size(ParallelMode.DATA))
        scaled_norm_tensor = torch.tensor(scaled_norm, device=get_current_device(), dtype=torch.float)
        dist.all_reduce(scaled_norm_tensor, group=pg)
        vocab_grad_norm = scaled_norm_tensor.item()

    # Scale.
    vocab_grad_norm[vocab_grad_norm == float("inf")] = -1
    vocab_grad_norm[vocab_grad_norm == -float("inf")] = -1
    vocab_grad_norm[torch.isnan(vocab_grad_norm)] = -2

    return vocab_grad_norm


def compute_param_metric(
    gradients,
    parameters,
    metric_type: str,
    last_stage=False,
    previous_param_metrics=None,
    norm_type=2,
    zero_mode=ParallelMode.ZERO1,
):
    """Get the metrics of params
    Argumemts:
        metric_type: (norm | zero_grad)
    """

    enable_cuda_kernels = gradients[0].device.type == "cuda"
    total_metrics = {}
    param_metrics = {}
    param_grads = reduce_grads(gradients, parameters, fine_grained=True)

    if metric_type == "norm":
        # Norm parameters.
        norm_type = float(norm_type)

    for param_name, grads in param_grads.items():
        if metric_type == "norm":
            param_metric = get_norm(grads, norm_type, enable_cuda_kernels)
            param_metrics[param_name] = param_metric.item() if torch.is_tensor(param_metric) else param_metric
        elif metric_type == "zero_grad":
            param_zero_grad_count = calc_zero_grad(grads)
            param_metrics[param_name] = param_zero_grad_count

    if last_stage is False:
        return param_metrics

    if previous_param_metrics is not None:
        for key, value in previous_param_metrics.items():
            if key not in param_metrics:
                param_metrics[key] = value
                continue
            if metric_type == "norm" and norm_type == inf:
                param_metrics[key] = max(param_metrics[key], value)
            else:
                param_metrics[key] += value

    # model parallel
    model_parallel_param_metrics = {}
    if gpc.is_initialized(ParallelMode.MODEL):
        parallel_param_metrics = [None for _ in range(gpc.get_world_size(ParallelMode.MODEL))]
        dist.all_gather_object(parallel_param_metrics, param_metrics, group=gpc.get_group(ParallelMode.MODEL))
        for local_param_metric in parallel_param_metrics:
            for param_name, param_metric in local_param_metric.items():
                if param_name not in model_parallel_param_metrics:
                    model_parallel_param_metrics[param_name] = 0.0
                if metric_type == "norm" and norm_type == inf:
                    model_parallel_param_metrics[param_name] = max(
                        model_parallel_param_metrics[param_name], param_metric
                    )
                else:
                    model_parallel_param_metrics[param_name] += param_metric

    # zero parallel
    zero_param_metrics = [None for _ in range(gpc.get_world_size(zero_mode))]
    dist.all_gather_object(zero_param_metrics, model_parallel_param_metrics, group=gpc.get_group(zero_mode))
    for local_param_metric in zero_param_metrics:
        for param_name, param_metric in local_param_metric.items():
            if param_name not in total_metrics:
                total_metrics[param_name] = 0.0
            if metric_type == "norm" and norm_type == inf:
                total_metrics[param_name] = max(total_metrics[param_name], param_metric)
            else:
                total_metrics[param_name] += param_metric

    # moe
    if zero_mode == ParallelMode.EXPERT_DATA:
        pg = gpc.get_group(ParallelMode.EXPERT)
        total_metric_values = list(total_metrics.values())
        if isinstance(total_metric_values[0], torch.Tensor):
            scaled_param_metric = torch.stack(total_metric_values).to(device=get_current_device())
        else:
            scaled_param_metric = torch.cuda.FloatTensor(total_metric_values, device=get_current_device())
        scaled_param_metric = scaled_param_metric / float(gpc.get_world_size(ParallelMode.EXPERT))
        dist.all_reduce(scaled_param_metric, group=pg)
        for i, param_name in enumerate(total_metrics.keys()):
            total_metrics[param_name] = scaled_param_metric[i]

    # calc zero grad percent
    if metric_type == "zero_grad":
        for param_name, param_metric in total_metrics.items():
            total_metrics[param_name] = (param_metric[0] / param_metric[1]).item()

    # scale norm
    if metric_type == "norm":
        for param_name, param_metric in total_metrics.items():
            if torch.is_tensor(param_metric):
                param_metric = param_metric.item()
            if param_metric in (inf, -inf):
                total_metrics[param_name] = -1
            elif math.isnan(param_metric):
                total_metrics[param_name] = -2
            else:
                total_metrics[param_name] = param_metric

    return total_metrics


def compute_param_norm(
    gradients,
    parameters,
    last_stage=False,
    previous_param_norms=None,
    norm_type=2,
    zero_mode=ParallelMode.ZERO1,
):
    """Get the norm of params
    Arguments:
        gradients (Iterable[Tensor]): The gradient value.
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        The norm of the parameters.
    """

    return compute_param_metric(
        gradients,
        parameters,
        metric_type="norm",
        last_stage=last_stage,
        previous_param_metrics=previous_param_norms,
        norm_type=norm_type,
        zero_mode=zero_mode,
    )


def compute_zero_grad_count(
    gradients,
    parameters,
    last_stage=False,
    previous_zero_grad_count=None,
    zero_mode=ParallelMode.ZERO1,
):
    """Get the count of zero gradient for each parameters
    Arguments:
        gradients (Iterable[Tensor]): The gradient value.
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to.

    Returns:
        The count of zero gradient for each parameters
    """

    return compute_param_metric(
        gradients,
        parameters,
        metric_type="zero_grad",
        last_stage=last_stage,
        previous_param_metrics=previous_zero_grad_count,
        zero_mode=zero_mode,
    )


def compute_layer_norm(param_norms, loss_scale):
    """
    compute layer norm by parameter norms
    """
    param_norms_groupby_layer = {}
    layer_norms = {}

    for param_name, param_norm in param_norms.items():
        layer_name, param_key = param_name.split("-")
        if param_key not in param_norms_groupby_layer:
            param_norms_groupby_layer[param_key] = {}
        if layer_name not in layer_norms:
            layer_norms[layer_name] = 0.0

        if param_norm not in (-1, -2):
            param_norm = param_norm**0.5 / loss_scale

        param_norms_groupby_layer[param_key][layer_name] = param_norm
        layer_norms[layer_name] += param_norm

    return layer_norms, param_norms_groupby_layer


def compute_layer_zero_grad_count(param_zero_grad_count):
    param_zero_grad_count_groupby_layer = {}
    layer_zero_grad_count = {}

    for param_name, zero_grad_count in param_zero_grad_count.items():
        layer_name, param_key = param_name.split("-")
        if param_key not in param_zero_grad_count_groupby_layer:
            param_zero_grad_count_groupby_layer[param_key] = {}
        if layer_name not in layer_zero_grad_count:
            layer_zero_grad_count[layer_name] = 0.0

        param_zero_grad_count_groupby_layer[param_key][layer_name] = zero_grad_count
        layer_zero_grad_count[layer_name] += zero_grad_count

    return layer_zero_grad_count, param_zero_grad_count_groupby_layer


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

            for _, children in _chunk.named_children():
                # should be the transformer block definaton in modeling_xxx.py
                if isinstance(children, nn.ModuleList):
                    # record the block that a parameter belongs to
                    for _, block in enumerate(children):
                        # self._block_to_param[f"{name}.{idx}"] = list(block.parameters())
                        self._block_to_param[block] = list(block.parameters())
                else:
                    # record the block that a parameter belongs to
                    # self._block_to_param[name] = list(children.parameters())
                    self._block_to_param[children] = list(children.parameters())

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
