#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Optional

import torch
from flash_attn.modules.embedding import ParallelGPT2Embeddings
from flash_attn.modules.mlp import ParallelFusedMLP
from torch import nn

from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.initialize.initialize_tensor import normal_, scaled_init_method_normal
from internlm.model.embedding import Embedding1D
from internlm.model.linear import (
    FeedForward,
    RewardModelLinear,
    ScaleColumnParallelLinear,
)
from internlm.model.multi_head_attention import MHA
from internlm.model.utils import gather_forward_split_backward, try_import_RMSNorm
from internlm.solver.pipeline_utils import partition_uniform
from internlm.utils.checkpoint import activation_checkpoint
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER

MODEL_TYPE = "INTERNLM"

logger = get_logger(__file__)
RMSNorm = try_import_RMSNorm()


class PackedFlashBaseLayer1D(nn.Module):
    """
    1D Packed Flash Base Layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
        use_flash_attn (bool): Whether use flash-attn. True by default.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        layer_idx: int = 0,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        norm_type: str = "rmsnorm",
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx
        self.use_flash_attn = use_flash_attn

        head_dim = hidden_size // num_attention_heads
        self.mixer = MHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            process_group=gpc.get_group(ParallelMode.TENSOR),
            dropout=attn_drop_rate,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            use_flash_attn=use_flash_attn,
            device=device,
            dtype=dtype,
        )

        self.dropout1 = nn.Dropout(drop_rate)
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(hidden_size, eps=layer_norm_epsilon)
            self.norm2 = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        if use_swiglu:
            self.mlp = FeedForward(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            self.mlp = ParallelFusedMLP(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                activation="gelu_approx",
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias1=False,
                bias2=False,
                sequence_parallel=gpc.config.parallel.sequence_parallel,
                checkpoint_lvl=0,
                heuristic="auto",
                device=device,
                dtype=dtype,
            )
        for _, param in self.mlp.named_parameters():
            if gpc.get_world_size(ParallelMode.TENSOR) > 1:
                setattr(param, IS_TENSOR_PARALLEL, True)
        self.dropout2 = nn.Dropout(drop_rate)
        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.mixer.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "Wqkv" in name:
                    normal_(std=0.006)(param.data)
                elif self.use_scaled_init:
                    scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                else:
                    normal_(std=0.0015)(param.data)

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1 and "bias" in name:
                    param.data.zero_()
                elif self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        normal_(std=0.006 if "w1" in name or "w2" in name else 0.0015)(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        normal_(std=0.006 if "fc1" in name else 0.0015)(param.data)

    def forward(self, hidden_states, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None):
        if self.checkpoint and self.training:
            return activation_checkpoint(
                self._forward, False, hidden_states, cu_seqlens, indexes, inference_params, max_seqlen
            )
        else:
            return self._forward(hidden_states, cu_seqlens, indexes, inference_params, max_seqlen)

    def _forward(self, hidden_states=None, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """
        mixer_kwargs = {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "indexes": indexes,
            "inference_params": inference_params,
        }

        def _dropout_and_norm_attn(_hidden_states):
            _dropped = self.dropout1(_hidden_states)
            _residual = _dropped
            _hidden_states = self.norm1(_residual.float())
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_attn(hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, **mixer_kwargs)

        def _dropout_and_norm_ffn(_residual, _hidden_states):
            _dropped = self.dropout2(_hidden_states)
            _residual = (_dropped + _residual) if _residual is not None else _dropped
            _hidden_states = self.norm2(_residual.float())
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_ffn, False, residual, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mlp(hidden_states)

        return hidden_states + residual


class PackedFlashInternLm1D(nn.Module):
    """
    1D Packed Flash InternLm.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        num_attention_heads (int): The number of attention head. 12 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 0.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    True by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.

    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        vocab_size: int = 50304,
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        checkpoint: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_split_hidden: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        device: Optional[torch.device] = None,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        checkpoint_layer_num = int(num_layers * checkpoint)

        if is_reward:
            head_cls = RewardModelLinear
        else:
            head_cls = ScaleColumnParallelLinear
        if first:
            if embed_split_hidden:
                self.embedding = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
            else:
                self.embedding = ParallelGPT2Embeddings(
                    embed_dim=hidden_size,
                    vocab_size=vocab_size,
                    max_position_embeddings=-1,
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    padding_idx=None,
                    sequence_parallel=gpc.config.parallel.sequence_parallel,
                    device=device,
                    dtype=dtype,
                )
            for _, param in self.embedding.named_parameters():
                normal_(std=0.0052)(param)
                if gpc.get_world_size(ParallelMode.TENSOR) > 1:
                    setattr(param, IS_TENSOR_PARALLEL, True)
        self.embed_grad_scale = embed_grad_scale
        self.blocks = nn.ModuleList(
            [
                PackedFlashBaseLayer1D(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    use_flash_attn=use_flash_attn,
                )
                for lid in range(num_layers)
            ]
        )
        if last:
            if norm_type == "rmsnorm":
                self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
            else:
                self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.head = head_cls(
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias=False,
                device=device,
                dtype=dtype,
                weight_scale=embed_grad_scale,
            )
            for _, param in self.head.named_parameters():
                normal_(std=0.0052)(param)
                if gpc.get_world_size(ParallelMode.TENSOR) > 1:
                    setattr(param, IS_TENSOR_PARALLEL, True)
        self.parallel_output = parallel_output

    def forward(self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "embedding"):
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )
        if isinstance(cu_seqlens, list):
            assert len(cu_seqlens) == 1
            cu_seqlens = cu_seqlens[0].to(hidden_states.device)

        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.squeeze(0)
            hidden_states = hidden_states.squeeze(0)  # If cu_seqlens is passed in，it indicated a packed state，
            # the batch dimension with a size of 1 should be directly squeezed off.

        if indexes is not None:
            assert len(indexes) == 1
            # The indexes are used to indicate the actual position IDs of each token in the packed input.
            indexes = indexes[0]
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item() if cu_seqlens is not None else None

        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                indexes=indexes,
                inference_params=inference_params,
                max_seqlen=max_seqlen,
            )

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())
        if hasattr(self, "head"):
            hidden_states = self.head(hidden_states)

        if not self.parallel_output:
            hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=-1)
        return hidden_states


def _build_generic_model_1d(num_layers, num_chunks, device=torch.device("cuda"), **kwargs):
    """
    build generic model 1d

    Args:
        num_layers (int): The number of layer.
        num_chunks (int): The number of partitions in pipeline parallel.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]
    if gpc.is_rank_for_log():
        logger.info(f"The layer sharding is {all_parts}.")

    models = []

    for start, end in parts:
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        # If there is no content in the final layer, assign the last layer.
        kwargs["last"] = end == num_layers and len(all_parts[-1]) != 0
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start
        chunk = PackedFlashInternLm1D(**filter_kwargs(PackedFlashInternLm1D.__init__, kwargs)).to(device)

        models.append(chunk)
    torch.distributed.barrier()
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model


@MODEL_INITIALIZER.register_module(module_name=MODEL_TYPE)
def build_model_with_cfg(
    num_chunks=1,
    checkpoint=0.0,
    dtype=torch.float,
    embed_split_hidden=False,
    num_layers=48,
    hidden_size=2048,
    vocab_size=50304,
    embed_grad_scale=1,
    parallel_output=True,
    num_attention_heads=32,
    mlp_ratio=4.0,
    residual_in_fp32=False,
    norm_type="rmsnorm",
    drop_rate=0,
    attn_drop_rate=0,
    apply_post_layer_norm=False,  # pylint: disable=W0613
    layer_norm_epsilon=1e-5,
    is_reward=False,
    dropout_selective_checkpoint=True,
    use_scaled_init: bool = True,
    use_swiglu: bool = True,
    use_flash_attn: bool = True,
):
    """
    Build model with config.

    Args:
        num_chunks (int): The number of partitions in pipeline parallel. 1 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. False by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    False by default.
        num_layers (int): The number of layer. 48 by default.
        hidden_size (int): The size of hidden state. 2048 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        num_attention_heads (int): The number of attention head. 32 by default.
        mlp_ratio (int): The ratio of MLP layers. 4.0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default. It cannot be used temporarily
                                 because this parameter requires inconsistent data types to be passed between pipelines,
                                 which requires significant modifications to internlm.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        drop_rate (float): The dropout rate of input hidden state. 0 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        apply_post_layer_norm (bool): Whether to apply post layer norm. False by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        is_reward (bool): Whether to use reward model. False by default.
        dropout_selective_checkpoint (bool): It can only be enabled when checkpoint is disabled. True by default.
        use_scaled_init (bool): Whether to use scaled init. True by default.
        use_swiglu (bool): Whether to use swiglu. True by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.

    """

    cfg = dict(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        checkpoint=checkpoint,
        dtype=dtype,
        embed_split_hidden=embed_split_hidden,
        vocab_size=vocab_size,
        embed_grad_scale=embed_grad_scale,
        parallel_output=parallel_output,
        mlp_ratio=mlp_ratio,
        residual_in_fp32=residual_in_fp32,
        norm_type=norm_type,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        is_reward=is_reward,
        dropout_selective_checkpoint=dropout_selective_checkpoint,
        use_scaled_init=use_scaled_init,
        use_swiglu=use_swiglu,
        use_flash_attn=use_flash_attn,
    )

    return _build_generic_model_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)
