#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import List

from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


class BaseStore:
    """
    Base Store
    """

    def __init__(self, dp_parallel_mode=ParallelMode.DATA):
        self._world_size = gpc.get_world_size(dp_parallel_mode)
        self._local_rank = gpc.get_local_rank(dp_parallel_mode)

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank


class BucketStore(BaseStore):
    """
    Bucket Store
    """

    def __init__(self, dp_parallel_mode):
        super().__init__(dp_parallel_mode)
        self._grads = dict()
        self._params = dict()
        self._num_elements_in_bucket = dict()

        self.reset()

    def num_elements_in_bucket(self, reduce_rank: int = None):
        return self._num_elements_in_bucket[reduce_rank]

    def add_num_elements_in_bucket(self, num_elements, reduce_rank: int = None):
        self._num_elements_in_bucket[reduce_rank] += num_elements

    def add_grad(self, tensor, reduce_rank: int = None):
        self._grads[reduce_rank].append(tensor)

    def add_param(self, tensor, reduce_rank: int = None):
        self._params[reduce_rank].append(tensor)

    def reset(self):
        keys = [None] + list(range(self._world_size))
        self._grads = {rank: [] for rank in keys}
        self._params = {rank: [] for rank in keys}
        self._num_elements_in_bucket = {rank: 0 for rank in keys}

    def reset_by_rank(self, reduce_rank=None):
        self._grads[reduce_rank] = []
        self._params[reduce_rank] = []
        self._num_elements_in_bucket[reduce_rank] = 0

    def get_grad(self, reduce_rank: int = None):
        return self._grads[reduce_rank]

    def get_param(self, reduce_rank: int = None):
        return self._params[reduce_rank]


class GradientStore(BaseStore):
    """
    Gradient Store
    """

    def __init__(self, *args):
        super().__init__(*args)
        # bookkeeping data structures
        self._averaged_gradients = dict()

        # for backward reduction hooks
        self._grad_acc_objs = []

    def add_accumulate_grad_object(self, obj):
        """
        Keep :class:`AccumulateGrad` objects. If these objects are not kept, reduction hooks may not
        be attached successfully.

        :param obj: An object of :class:`AccumulateGrad` class
        :type obj: :class:`AccumulateGrad`
        """

        self._grad_acc_objs.append(obj)

    def get_averaged_gradients_by_group(self, group_id: int) -> List[Tensor]:
        """
        Return average gradients of a parameter group

        :param group_id: The index of parameter group
        :type group_id: int

        :return: Return the list of averaged gradients of a parameter group. Each element is a gradient,
            not a parameter.
        :rtype: List[torch.Tensor]
        """

        return self._averaged_gradients[group_id]

    def add_average_gradient_by_group(self, group_id: int, tensor: Tensor) -> None:
        """
        Append an average gradient to the list of averaged gradients of a parameter group

        :param group_id: The index of a parameter group
        :param tensor: A :class:`torch.Tensor` object
        :type group_id: int
        :type tensor: torch.Tensor

        """

        if group_id in self._averaged_gradients:
            self._averaged_gradients[group_id].append(tensor)
        else:
            self._averaged_gradients[group_id] = [tensor]

    def reset_average_gradients_by_group(self, group_id: int) -> None:
        """
        Reset the bookkeeping data structure for averaged gradients to an empty list

        :param group_id: The index of a parameter group
        :type group_id: int
        """

        self._averaged_gradients[group_id] = []


class ParameterStore(BaseStore):
    """
    Parameter Store
    """

    def __init__(self, dp_paralle_mode):
        super().__init__(dp_paralle_mode)
        # param partitioning data structures
        self._fp16_param_to_rank = dict()
        self._rank_groupid_to_fp16_param_list = dict()
        self._rank_group_id_to_flat_fp16_param = dict()

        # param reduction data structures
        self._is_param_reduced = dict()
        self._reduced_param = []

        self._former_bucket_reduced_param = {}
        self._last_bucket_reduced_param = {}
        self._former_bucket_reduced_grad = {}
        self._last_bucket_reduced_grad = {}

    def set_param_to_rank(self, tensor: Tensor, rank: int) -> None:
        """
        Set the mapping between parameter to rank, each parameter should be owned by a rank.

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor
        :param rank: The rank of which the process is responsible for updating the parameter
        :type rank: int
        """

        self._fp16_param_to_rank[tensor] = rank

    def get_param_rank(self, tensor: Tensor) -> int:
        """
        Gives the rank which the parameter belongs to

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor
        """
        return self._fp16_param_to_rank[tensor]

    def belongs_to_current_rank(self, tensor) -> bool:
        """
        Check whether a parameter is supposed to be updated by the process of the current rank

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor

        :return: True if the parameter should be updated by the current rank. Otherwise false.
        :rtype: bool
        """

        tensor_rank = self._fp16_param_to_rank[tensor]
        return tensor_rank == self._local_rank

    def add_fp16_param_list_by_rank_group(self, rank, group_id, tensor_list) -> None:
        if rank not in self._rank_groupid_to_fp16_param_list:
            self._rank_groupid_to_fp16_param_list[rank] = dict()

        if group_id not in self._rank_groupid_to_fp16_param_list[rank]:
            self._rank_groupid_to_fp16_param_list[rank][group_id] = []

        self._rank_groupid_to_fp16_param_list[rank][group_id].extend(tensor_list)

    def get_fp16_params_by_rank_group(self, rank, group_id) -> List[Tensor]:
        return self._rank_groupid_to_fp16_param_list[rank][group_id]

    def add_flat_fp16_param_by_rank_group(self, rank, group_id, tensor) -> None:
        if rank not in self._rank_group_id_to_flat_fp16_param:
            self._rank_group_id_to_flat_fp16_param[rank] = dict()

        self._rank_group_id_to_flat_fp16_param[rank][group_id] = tensor

    def get_flat_fp16_param_by_rank_group(self, rank, group_id) -> Tensor:
        return self._rank_group_id_to_flat_fp16_param[rank][group_id]

    def is_param_reduced(self, tensor):
        return self._is_param_reduced[tensor]

    def set_param_reduction_state(self, tensor, state):
        self._is_param_reduced[tensor] = state

    def get_param_reduction_states(self):
        return self._is_param_reduced

    def reset_previous_reduced_params(self):
        self._reduced_param = []

    def add_previous_reduced_param(self, tensor):
        self._reduced_param.append(tensor)

    def add_reduced_param_for_compute_norm(self, param, last_bucket=False):
        group_id = getattr(param, "group_id")
        if last_bucket:
            if group_id not in self._last_bucket_reduced_param:
                self._last_bucket_reduced_param[group_id] = []
                self._last_bucket_reduced_grad[group_id] = []

            self._last_bucket_reduced_param[group_id].append(param)
            self._last_bucket_reduced_grad[group_id].append(param.grad)
        else:
            if group_id not in self._former_bucket_reduced_param:
                self._former_bucket_reduced_param[group_id] = []
                self._former_bucket_reduced_grad[group_id] = []

            self._former_bucket_reduced_param[group_id].append(param)
            self._former_bucket_reduced_grad[group_id].append(param.grad)

    def get_reduced_param_for_compute_norm(self, group_id=0, last_bucket=False):
        if not last_bucket:
            if group_id not in self._former_bucket_reduced_param:
                return [], []
            return (
                self._former_bucket_reduced_param[group_id],
                self._former_bucket_reduced_grad[group_id],
            )
        else:
            if group_id not in self._last_bucket_reduced_param:
                return [], []
            return (
                self._last_bucket_reduced_param[group_id],
                self._last_bucket_reduced_grad[group_id],
            )

    def reset_reduced_data_for_compute_norm(self):
        self._former_bucket_reduced_param = {}
        self._last_bucket_reduced_param = {}
        self._former_bucket_reduced_grad = {}
        self._last_bucket_reduced_grad = {}

    def clear_grads_of_previous_reduced_params(self):
        if len(self._reduced_param) > 0:
            for param in self._reduced_param:
                param.grad = None
            self.reset_previous_reduced_params()


class TensorBucket:
    """
    Tensor Bucket
    """

    def __init__(self, size):
        self._max_size = size
        self._current_size = 0
        self._bucket = []
        self._flat_tensor = None
        self._unflatten_and_copy_flag = False
        self.commu_handle = None

    @property
    def max_size(self):
        return self._max_size

    @property
    def current_size(self):
        return self._current_size

    def is_full_or_oversized(self):
        return self._current_size >= self._max_size

    def is_empty(self):
        return len(self._bucket) == 0

    def set_unflatten_and_copy_flag(self, flag):
        self._unflatten_and_copy_flag = flag

    def get_unflatten_and_copy_flag(self):
        return self._unflatten_and_copy_flag

    def get_flat_tensor(self):
        return self._flat_tensor

    def add_to_bucket(self, tensor, allow_oversize=False):
        tensor_size = tensor.numel()

        if not allow_oversize and self.will_exceed_max_size(tensor_size):
            msg = f"The param bucket max size {self._max_size} is exceeded" + f"by tensor (size {tensor_size})"
            raise RuntimeError(msg)

        self._bucket.append(tensor)
        self._current_size += tensor_size

    def will_exceed_max_size(self, tensor_size):
        expected_size = self._current_size + tensor_size
        return expected_size > self._max_size

    def get_bucket(self):
        return self._bucket

    def empty(self):
        self._bucket = []
        self._size = 0
        self._flat_tensor = None
        self.commu_handle = None

    def flatten(self):
        self._flat_tensor = _flatten_dense_tensors(self._bucket)

    def unflatten_and_copy(self):
        if self._unflatten_and_copy_flag:
            unflattened_tensor_list = _unflatten_dense_tensors(self._flat_tensor, self._bucket)
            for old, new in zip(self._bucket, unflattened_tensor_list):
                old.copy_(new)
