#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from internlm.core.context import Config, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.moe import is_moe_param
from internlm.monitor import send_alert_message
from internlm.solver.optimizer.store import (
    BucketStore,
    GradientStore,
    ParameterStore,
    TensorBucket,
)
from internlm.solver.optimizer.utils import (
    DynamicGradScaler,
    flatten,
    get_grad_accumulate_object,
    has_inf_or_nan,
    reduce_tensor,
    release_param_grad,
    split_half_float_double,
    sync_param,
)
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer

from .utils import compute_norm

logger = get_logger(__file__)


class BaseOptimizer(Optimizer):
    """
    Base Optimizer.
    """

    def __init__(self, optim: Optimizer):  # pylint: disable=W0231
        self.optim = optim

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.optim.zero_grad(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.optim.load_state_dict(*args, **kwargs)

    def state_dict(self):
        return self.optim.state_dict()

    def backward(self, loss):
        loss.backward()

    def backward_by_grad(self, tensor, grad):
        torch.autograd.backward(tensors=tensor, grad_tensors=grad)

    def clip_grad_norm(self):
        pass


class HybridZeroOptimizer(BaseOptimizer):
    """
    Hybrid Zero Optimizer.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        cpu_offload=False,
        overlap_broadcast=False,
        grad_scal_cfg: Config = None,
        zero_cfg: Config = None,
        has_moe: bool = False,
    ):
        # DynamicGradScaler related args
        if gpc.config.model.dtype is torch.float32:
            initial_scale = 1
        else:
            initial_scale = grad_scal_cfg.fp16.initial_scale
        min_scale = grad_scal_cfg.fp16.min_scale
        growth_interval = grad_scal_cfg.fp16.growth_interval
        growth_factor = grad_scal_cfg.growth_factor
        backoff_factor = grad_scal_cfg.backoff_factor
        hysteresis = grad_scal_cfg.hysteresis
        max_scale = grad_scal_cfg.max_scale

        # Zero related args
        overlap_communication = zero_cfg.zero_overlap_communication
        reduce_bucket_size = zero_cfg.reduce_bucket_size
        clip_grad_norm = zero_cfg.clip_grad_norm

        super().__init__(optim=optimizer)

        self.has_moe = has_moe

        self._dtype = self.optim.param_groups[0]["params"][0].dtype
        self._cpu_offload = cpu_offload
        self._zero_local_rank = gpc.get_local_rank(ParallelMode.ZERO1)
        self._zero_world_size = gpc.get_world_size(ParallelMode.ZERO1)
        self._broadcast_parallel_mode = ParallelMode.ZERO1

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(ParallelMode.ZERO1)
        self._grad_store = GradientStore(ParallelMode.DATA)
        self._bucket_store = BucketStore(ParallelMode.DATA)

        # fp16 and fp32 params for mixed precision training
        self._fp16_param_groups = dict()
        self._fp32_flat_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size

        # gradient scaler
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
        )
        self._found_overflow = torch.cuda.FloatTensor([0], device=get_current_device())

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        # need to record the rank in which parameter groups are not assigned parameters.
        self.param_group_has_params = []
        self.param_group_no_params_ranks = []
        self.padding_grad = torch.zeros([32], dtype=self._dtype, device=get_current_device())
        self.padding_tensor = torch.zeros([32], dtype=self._dtype, device=get_current_device())

        self.rank_unique_id = (
            f"gpus-{gpc.get_world_size(ParallelMode.GLOBAL)}_"
            + f"pp-{gpc.get_local_rank(ParallelMode.PIPELINE)}_"
            + f"tp-{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"zo-{self._zero_local_rank}.pt"
        )
        self.params_per_rank_id_dict = []
        self.overlap_broadcast = overlap_broadcast

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = param_group["params"]

            # add the fp16 params to fp16_param_groups for bookkeeping
            self._fp16_param_groups[group_id] = group_params

            # assign parameters to ranks the params in the list are sorted
            params_per_rank, no_params_ranks = self._partition_param_list(group_params)
            self.param_group_no_params_ranks.append(no_params_ranks)
            self.param_group_has_params.append(self._zero_local_rank not in no_params_ranks)

            # store the mapping between param to rank each param should belong to only one rank
            for rank, params in enumerate(params_per_rank):
                # check whether any rank is not assigned params.
                if len(params) != 0:
                    self._param_store.add_fp16_param_list_by_rank_group(rank, group_id, params)
                    for param in params:
                        self._param_store.set_param_to_rank(param, rank)

            # move to cpu to make room to create the flat tensor
            for param in group_params:
                param.data = param.data.cpu()

            # flatten the reordered tensors
            for rank in range(self._zero_world_size):
                # No flat fp16 buffer is allocated if the process has no parameters.
                if rank not in self.param_group_no_params_ranks[group_id]:
                    tensor_list = self._param_store.get_fp16_params_by_rank_group(rank, group_id)
                    with torch.no_grad():
                        flat_tensor = flatten(tensor_list)
                    flat_tensor = flat_tensor.data.cuda()
                    self._param_store.add_flat_fp16_param_by_rank_group(rank, group_id, flat_tensor)
                    sync_param(flat_tensor=flat_tensor, tensor_list=tensor_list)

            # create a copy of fp32 weights of the parameters for which this rank is responsible
            # No flat fp32 buffer is allocated if the process has no parameters.
            if self.param_group_has_params[group_id]:
                fp16_flat_current_rank = self._param_store.get_flat_fp16_param_by_rank_group(
                    self._zero_local_rank, group_id
                )
                fp32_flat_current_rank = fp16_flat_current_rank.float()
                device = "cpu" if self._cpu_offload else get_current_device()
                fp32_flat_current_rank = fp32_flat_current_rank.to(device)
                fp32_flat_current_rank.requires_grad = True
                self._fp32_flat_param_groups_of_current_rank[group_id] = fp32_flat_current_rank

                # need to replace the params in the `params` field in the optimizer
                # so that when the optimizer calls step(), it only updates the tensors
                # managed by this data parallel rank
                param_group["params"] = [fp32_flat_current_rank]

            # set reduction state
            for param in self._fp16_param_groups[group_id]:
                self._param_store.set_param_reduction_state(param, False)

        assert len(self._fp16_param_groups) != 0

        # If a rank is not assigned any arguments, 'has_params' is False.
        self.has_params = sum(self.param_group_has_params) != 0
        # flag used to skip unnecessary gradient reduce operation when gradient accumulation is enabled.
        self.skip_grad_reduce = False

        # initialize communication stream for
        # communication-computation overlapping
        if self._overlap_communication:
            self._comm_stream = torch.cuda.Stream()

        # reduction hook is only used if overlapping communication
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication:
            self._attach_reduction_hook()

    @property
    def zero_local_rank(self):
        return self._zero_local_rank

    @property
    def zero_world_size(self):
        return self._zero_world_size

    @property
    def dtype(self):
        return self._dtype

    @property
    def loss_scale(self):
        return self.grad_scaler.scale

    @property
    def num_param_groups(self):
        return len(self._fp16_param_groups)

    def _partition_param_list(self, param_list):
        no_params_ranks = []
        params_per_rank = [[] for _ in range(self._zero_world_size)]
        numel_per_rank = [0 for _ in range(self._zero_world_size)]
        self.params_per_rank_id_dict.append([[] for _ in range(self._zero_world_size)])

        sorted_params = sorted(param_list, key=lambda x: x.numel(), reverse=True)
        for i, param in enumerate(sorted_params):
            global_id = str(i)
            for j in range(len(param.size())):
                global_id = "_".join([global_id, str(param.size()[j])])

            rank_to_go = numel_per_rank.index(min(numel_per_rank))
            params_per_rank[rank_to_go].append(param)
            self.params_per_rank_id_dict[-1][rank_to_go].append(global_id)
            numel_per_rank[rank_to_go] += param.numel()

        # check whether any rank is not assigned to parameters.
        for rank, params in enumerate(params_per_rank):
            if len(params) == 0:
                no_params_ranks.append(rank)

        if gpc.is_rank_for_log():
            logger.info(  # pylint: disable=W1203
                f"Number of elements on ranks: {numel_per_rank}, rank:{gpc.get_global_rank()}"
            )

        return params_per_rank, set(no_params_ranks)

    def _attach_reduction_hook(self):
        # we iterate over the fp16 params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._fp16_param_groups[group_id]
            for param in param_group:
                if param.requires_grad and not is_moe_param(param):
                    reduce_rank = None

                    def _define_and_attach(param, reduce_rank=None):
                        # get the AccumulateGrad object of the param itself
                        # If these objects are not kept, reduction hooks may not be attached successfully.
                        accum_grad_obj = get_grad_accumulate_object(param)
                        self._grad_store.add_accumulate_grad_object(accum_grad_obj)

                        reduction_func = partial(
                            self._store_and_try_reduce_grads_by_bucket, param=param, reduce_rank=reduce_rank
                        )

                        # define hook
                        # NOT IMPORTANT BUT GOOD TO KNOW:
                        # args here is not grad, but allow_unreacable and accumulate_grad
                        def reduce_grad_hook(*args):  # pylint: disable=W0613
                            if self.skip_grad_reduce is False:
                                reduction_func()

                        accum_grad_obj.register_hook(reduce_grad_hook)

                    _define_and_attach(param, reduce_rank)

    def _store_and_try_reduce_grads_by_bucket(self, param, reduce_rank=None):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # after reduction, the bucket will be empty
        if self._bucket_store.num_elements_in_bucket(reduce_rank) + param_size > self._reduce_bucket_size:
            self._reduce_grads_stored_in_bucket(reduce_rank)

        # the param must not be reduced to ensure correctness
        is_param_reduced = self._param_store.is_param_reduced(param)
        if is_param_reduced:
            msg = (
                f"Parameter of size ({param.size()}) has already been reduced, "
                + "duplicate reduction will lead to arithmetic incorrectness"
            )
            raise RuntimeError(msg)

        # the param must have grad for reduction
        assert param.grad is not None, f"Parameter of size ({param.size()}) has None grad, cannot be reduced"

        self._bucket_store.add_num_elements_in_bucket(param_size, reduce_rank)
        self._bucket_store.add_grad(param.grad, reduce_rank)
        self._bucket_store.add_param(param, reduce_rank)

    def _reduce_grads_stored_in_bucket(self, reduce_rank=None):
        # reduce grads
        self._reduce_grads_by_rank(
            reduce_rank=reduce_rank,
            grads=self._bucket_store.get_grad(reduce_rank=reduce_rank),
            bucket_size=self._bucket_store.num_elements_in_bucket(reduce_rank),
        )

        # use communication stream if overlapping
        # communication with computation
        if self._overlap_communication:
            stream = self._comm_stream
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            params_in_bucket = self._bucket_store.get_param(reduce_rank=reduce_rank)

            for param in params_in_bucket:
                # the is_param_reduced flag should be False showing that
                # this param is not reduced before calling self._reduce_grads_by_rank
                is_param_reduced = self._param_store.is_param_reduced(param)

                if is_param_reduced:
                    msg = (
                        f"Parameter of size ({param.size()}) has been reduced, "
                        + "duplicate reduction will lead to arithmetic incorrectness"
                    )
                    raise RuntimeError(msg)

                # update the flag
                self._param_store.set_param_reduction_state(param, True)

        self._bucket_store.reset_by_rank(reduce_rank)

    def _reduce_grads_by_rank(self, reduce_rank, grads, bucket_size):
        grad_buckets_by_dtype = split_half_float_double(grads)

        for tensor_list in grad_buckets_by_dtype:
            param_bucket = TensorBucket(size=bucket_size)
            for tensor in tensor_list:
                param_bucket.add_to_bucket(tensor, allow_oversize=True)
                if param_bucket.is_full_or_oversized():
                    self._reduce_and_copy(bucket=param_bucket, reduce_rank=reduce_rank)
                    param_bucket.empty()
            if not param_bucket.is_empty():
                self._reduce_and_copy(bucket=param_bucket, reduce_rank=reduce_rank)

    def _reduce_and_copy(self, bucket: TensorBucket, reduce_rank):
        if self._overlap_communication:
            torch.cuda.synchronize()
            self._param_store.clear_grads_of_previous_reduced_params()
            stream = self._comm_stream
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            flat = bucket.flatten()
            reduced_flat = reduce_tensor(
                tensor=flat, dtype=self.dtype, dst_rank=reduce_rank, parallel_mode=ParallelMode.DATA
            )

            # update the reduced tensor
            if reduce_rank is None or reduce_rank == self._zero_local_rank:
                bucket.unflatten_and_copy(reduced_flat)

    def _has_inf_or_nan(self, tensor):
        try:
            tensor_mean = float(tensor.mean())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if tensor_mean == float("inf") or tensor_mean == -float("inf"):
                return True
            return False

    def _sync_grad(self):
        # update param already reduced flag
        reduction_states = self._param_store.get_param_reduction_states()
        for tensor, _ in reduction_states.items():
            reduction_states[tensor] = False

        # accumulate gradient
        avg_gradients = self._grad_store._averaged_gradients
        for group_id in range(self.num_param_groups):
            # the following operations are performed only on the rank to which parameters are assigned.
            if self._zero_local_rank not in self.param_group_no_params_ranks[group_id]:
                param_group = self._param_store.get_fp16_params_by_rank_group(self._zero_local_rank, group_id)

                if group_id not in avg_gradients:
                    avg_gradients[group_id] = []

                param_idx = 0
                for param in param_group:
                    if param.grad is not None:
                        if len(avg_gradients[group_id]) == param_idx:
                            avg_gradients[group_id].append(param.grad)
                        else:
                            avg_gradients[group_id][param_idx].add_(param.grad)
                        param_idx += 1

        # the gradients needed are stored in the avg_gradients buffer
        # thus, can clear this
        self.zero_grad()

    def zero_grad(self, set_to_none=True):
        """
        Set parameter gradients to zero. If set_to_none = True, gradient
        will be set to None to save memory.

        :param set_to_none: Whether set the gradient to None. Default value is True.
        :type set_to_none: bool
        """
        for _, param_group in self._fp16_param_groups.items():
            for param in param_group:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.detach()
                    param.grad.zero_()
                else:
                    pass

    def backward(self, loss, retain_graph=False):
        loss = self.loss_scale * loss
        loss.backward(retain_graph=retain_graph)

        # Gradients may not be fully synchronized here.

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            Union[bool, float]: Whether the gradient is success updated, and the gradient.
        """
        assert closure is None, "closure is not supported by step()"

        timer("sync_grad").start()
        # if not overlapping communication (no reduction hook is attached)
        # we need to manually reduce these gradients
        if not self._overlap_communication:
            for group_id in range(len(self._fp16_param_groups)):
                for param in self._fp16_param_groups[group_id]:
                    if param.grad is not None:
                        self._store_and_try_reduce_grads_by_bucket(param)

        # we need to reduce the gradients left in the communication bucket
        self._reduce_grads_stored_in_bucket()

        # clear reduced grads
        if self._overlap_communication:
            torch.cuda.synchronize()
            self._param_store.clear_grads_of_previous_reduced_params()

        self._sync_grad()
        timer("sync_grad").stop()

        return self._step(closure=closure)

    def _get_norm_with_moe_layers(self, norm_groups):
        # all_groups_norm_old = all_groups_norm
        # Need to allreduce(avg) the norms across different ranks because moe params will not be synced during allreduce
        pg = gpc.get_group(ParallelMode.DATA)
        print(type(norm_groups))
        scaled_norm = norm_groups * 1.0 / float(gpc.get_world_size(ParallelMode.DATA))
        scaled_norm_tensor = torch.tensor(
            scaled_norm, device=self._fp32_flat_param_groups_of_current_rank[0].device, dtype=torch.float
        )
        dist.all_reduce(scaled_norm_tensor, group=pg)
        all_groups_norm = scaled_norm_tensor.item()
        # print(f"old = {all_groups_norm_old} and new = {all_groups_norm} at rank: {deepspeed.comm.get_rank()}")
        return all_groups_norm

    def _step(self, closure=None):
        assert closure is None, "closure is not supported by step()"

        # check for overflow
        found_inf = self._check_overflow()
        # Because you may encounter inf when computing norm
        timer("cal_norm").start()
        norm_groups = []
        for group_id in range(self.num_param_groups):
            # compute norm
            if self._zero_local_rank not in self.param_group_no_params_ranks[group_id]:
                gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
                parameters = self._param_store.get_fp16_params_by_rank_group(
                    group_id=group_id, rank=self._zero_local_rank
                )
            else:
                # in order to prevent collection communication from hanging,
                # we need to involve rank that are not assigned parameters in compute_norm(),
                # so we give them a fp16 vector of 0 values.
                gradients = [self.padding_grad]
                parameters = [self.padding_tensor]

            if self._clip_grad_norm > 0:
                # this norm is before scaling, it will be very large
                norm_group = compute_norm(
                    gradients=gradients,
                    parameters=parameters,
                )
                if norm_group == -1:
                    timer("cal_norm").stop()
                    found_inf = True
                    break
                norm_groups.append(norm_group)

        loss_scale = float(self.loss_scale.item())  # backup
        if gpc.config.model.dtype is not torch.float32:
            self.grad_scaler.update(found_inf)
        # update loss scale if overflow occurs
        if found_inf:
            if gpc.is_rank_for_log():
                logger.warning("Overflow occurs, please check it.")
                send_alert_message(address=gpc.config.alert_address, message="Overflow occurs, please check it.")
            self._grad_store._averaged_gradients = dict()
            self.zero_grad()
            return False, None

        # copy the grad of fp16 param to fp32 param
        single_grad_partition_groups = []
        global_norm = 0
        for group_id in range(self.num_param_groups):
            # compute norm
            # The following operations are performed only on the rank to which parameters are assigned.
            if not self.param_group_has_params[group_id]:
                continue
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)

            # create flat gradient for the flat fp32 params
            fp16_avg_grads = gradients
            flat_fp16_avg_grads = flatten(fp16_avg_grads)

            dtype = self._fp32_flat_param_groups_of_current_rank[group_id].dtype
            flat_fp32_avg_grads = flat_fp16_avg_grads.to(dtype)

            param_shape = self._fp32_flat_param_groups_of_current_rank[group_id].shape
            assert (
                param_shape == flat_fp32_avg_grads.shape
            ), f"fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}"

            single_grad_partition_groups.append(flat_fp32_avg_grads)
            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[group_id].grad = flat_fp32_avg_grads.to(device)
            self._grad_store._averaged_gradients[group_id] = []
            self._grad_store._averaged_gradients[group_id] = []

        # unscale and clip grads
        # get the global norm
        if self._clip_grad_norm > 0:
            global_norm = sum(norm_groups) ** 0.5

        if self.has_moe:
            global_norm = self._get_norm_with_moe_layers(global_norm)

        # the following operations are performed only on the rank to which parameters are assigned.
        if gpc.config.model.dtype is not torch.float32:
            if len(single_grad_partition_groups) != 0:
                self._unscale_and_clip_grads(single_grad_partition_groups, global_norm, loss_scale)

        timer("cal_norm").stop()
        # update the parameters
        timer("step").start()

        # For those ranks that are not assigned parameters, we just wait for other ranks
        # to send them updated their own parameters.
        if self.has_params:
            self.optim.step()
            # release the fp32 grad
            release_param_grad(self._fp32_flat_param_groups_of_current_rank.values())
            # update fp16 partition updated by the current rank
            for group_id in range(len(self._fp16_param_groups)):
                if self.param_group_has_params[group_id]:
                    fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                        rank=self._zero_local_rank, group_id=group_id
                    )
                    fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                    fp16_param.data.copy_(fp32_param)

        # TODO: support broadcast overlap
        self.broadcast_params(overlap=False)

        timer("step").stop()
        # update gradients may not be needed here, because the sync_params function is used in initialization,
        # so synchronization is maintained
        return True, global_norm / loss_scale

    def broadcast_params(self, overlap=False):
        handles = []

        for group_id in range(self.num_param_groups):
            for rank in range(self._zero_world_size):
                # The following operations are performed only on the rank to which parameters are assigned.
                if rank not in self.param_group_no_params_ranks[group_id]:
                    fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=rank, group_id=group_id)
                    # grank = gpc.get_ranks_in_group(group_type)[rank]  # need to convert to the global rank
                    # assert grank == rank, f"{grank} == {rank}"
                    g_rank = gpc.get_ranks_in_group(self._broadcast_parallel_mode)[rank]
                    handle = dist.broadcast(
                        fp16_param, src=g_rank, group=gpc.get_group(ParallelMode.ZERO1), async_op=True
                    )
                    handles.append(handle)

        if not overlap:
            for handle in handles:
                handle.wait()
        else:
            return handles

    ##################
    # FP16 Utilities #
    ##################

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group_id in range(len(self._fp16_param_groups)):
            # The following operations are performed only on the rank to which parameters are assigned.
            if self._zero_local_rank not in self.param_group_no_params_ranks[group_id]:
                for avg_grad in self._grad_store.get_averaged_gradients_by_group(group_id):
                    if avg_grad is not None and has_inf_or_nan(avg_grad):
                        self._found_overflow.fill_(1.0)
                        break
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.GLOBAL))

        return self._found_overflow.item() > 0

    def _unscale_and_clip_grads(self, grad_groups_flat, total_norm, loss_scale):
        # compute combined scale factor for this group
        combined_scale = loss_scale

        if self._clip_grad_norm > 0.0:
            # norm is in fact norm*scale
            clip = ((total_norm / loss_scale) + 1e-6) / self._clip_grad_norm
            if clip > 1.0:
                combined_scale = clip * loss_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1.0 / combined_scale)

    def clip_grad_norm(self, model, max_norm):
        # will conduct in the step()
        pass

    def state_dict(self):
        states = {}
        grad_scaler = self.grad_scaler.state_dict()
        states["grad_scaler"] = grad_scaler
        optim_states = self.optim.state_dict()
        states["base_optim_states"] = optim_states

        flat_fp32_weights = {}
        for group_id, param in self._fp32_flat_param_groups_of_current_rank.items():
            if self._zero_local_rank not in self.param_group_no_params_ranks[group_id]:
                assert param.grad is None
                flat_fp32_weights[group_id] = param
        states["flat_fp32_weights"] = flat_fp32_weights
        states["zero_devide_optim_plan"] = self.params_per_rank_id_dict

        return states

    def load_state_dict(self, states):
        # TODO: Need to take into account the change in the number of DP.
        assert "grad_scaler" in states, "Not found grad_scaler state!"
        grad_scaler = states["grad_scaler"]
        self.grad_scaler.load_state_dict(grad_scaler)
        optim_states = states["base_optim_states"]
        self.optim.load_state_dict(optim_states)

        # load fp32 model weight.
        flat_fp32_weights = states["flat_fp32_weights"]
        assert set(flat_fp32_weights.keys()) == set(self._fp32_flat_param_groups_of_current_rank)
        for group_id, param in flat_fp32_weights.items():
            if self._zero_local_rank not in self.param_group_no_params_ranks[group_id]:
                self_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                assert (
                    self_param.shape == param.shape
                ), f"The loaded parameter shape is inconsistent, {self_param.shape} != {param.shape}"
                self_param.data.copy_(param.data)

        # Load the fp16 model weights.
        for group_id in range(len(self._fp16_param_groups)):
            if self._zero_local_rank not in self.param_group_no_params_ranks[group_id]:
                fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                    rank=self._zero_local_rank, group_id=group_id
                )
                fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                fp16_param.data.copy_(fp32_param)

        if "zero_devide_optim_plan" in states:
            self.params_per_rank_id_dict = states["zero_devide_optim_plan"]
