"""
fair llama based on flash attention
Refer to https://github.com/facebookresearch/llama.git
"""

import math
import os
import sys
from typing import Optional, Union

import torch
from flash_attn.modules.embedding import ParallelGPT2Embeddings
from flash_attn.modules.mlp import ParallelFusedMLP
from flash_attn.ops.layer_norm import dropout_add_layer_norm
from torch import nn

from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.initialize.initialize_tensor import normal_, scaled_init_method_normal
from internlm.model.embedding import Embedding1D
from internlm.model.linear import RewardModelLinear, ScaleColumnParallelLinear
from internlm.model.multiquery_llama.modules import (
    FeedForward,
    OneDConvertedParallelMHA2,
)
from internlm.model.utils import gather_forward_split_backward, try_import_RMSNorm
from internlm.solver.pipeline_utils import partition_uniform_with_embed
from internlm.utils.checkpoint import activation_checkpoint
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


MODEL_TYPE = "MQLLAMA"
logger = get_logger(__file__)


def xavier_normal_tensor_parallel(tensor, partition_dim, gain=1, tp_degree=1):
    """initialize distributed tensor"""
    assert len(tensor.shape) == 2
    fan_in, fan_out = tensor.shape
    if partition_dim == 0:
        fan_in *= tp_degree
    else:
        fan_out *= tp_degree

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return torch.nn.init._no_grad_normal_(tensor, 0, std)


class PackedFlashConvertedLLAMALayer1D2(nn.Module):
    """
    1D Packed Flash Converted LLAMA Layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        apply_post_layer_norm (bool): Whether to use postnorm(True) or prenorm(False). False by default.
        fused_dropout_add_ln (bool): Whether to use dropout_add_layer_norm. True by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        no_bias (bool): Whether to use bias in multihead attention and feedforward. False by default
        deepnorm (bool): Whether to use deepnorm or not. False by default
        total_layers (int): The total numer of layers. Needed when using deepnorm. -1 by default.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.

    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_kv_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-5,
        apply_post_layer_norm: bool = False,
        fused_dropout_add_ln: bool = True,
        checkpoint: bool = False,
        layer_idx: int = 0,
        residual_in_fp32: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        no_bias: bool = False,
        deepnorm: bool = False,
        total_layers: int = -1,
        norm_type: str = "rmsnorm",
        adapt_hf: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.prenorm = not apply_post_layer_norm
        assert not fused_dropout_add_ln, "dropout_add_layer_norm can not be used here"
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.total_layers = total_layers
        self.layer_idx = layer_idx
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False

        if deepnorm:
            assert total_layers != -1, "When using deepnorm must pass in the total numer of layers"
            self.deepnorm_alpha = (2 * total_layers) ** 0.5  # refer to the code of GLM-130B
            self.deepnorm_beta = (
                2 * total_layers
            ) ** -0.5  # from LargeScale.megatron.model.utils:get_deepnorm_coefficients
            # refer to: https://kexue.fm/archives/8978
        else:
            self.deepnorm_alpha = 1.0
            self.deepnorm_beta = 1.0

        head_dim = hidden_size // num_attention_heads
        self.attention = OneDConvertedParallelMHA2(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_attention_heads,
            process_group=gpc.get_group(ParallelMode.TENSOR),
            bias=not no_bias,
            dropout=attn_drop_rate,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            use_flash_attn=True,
            checkpointing=False,
            device=device,
            dtype=dtype,
            rot_embed_HF_impl=adapt_hf,
        )

        self.adapt_hf = adapt_hf
        self.dropout1 = nn.Dropout(drop_rate)
        if norm_type == "rmsnorm":
            RMSNorm = try_import_RMSNorm()
            self.attention_norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        else:
            self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        if use_swiglu:
            self.feed_forward = FeedForward(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias=not no_bias,
                device=device,
                dtype=dtype,
            )
        else:
            self.feed_forward = ParallelFusedMLP(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                activation="gelu_approx",
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias1=not no_bias,
                bias2=not no_bias,
                sequence_parallel=gpc.config.model.sequence_parallel,
                checkpoint_lvl=0,
                heuristic="auto",
                device=device,
                dtype=dtype,
            )
        self.use_swiglu = use_swiglu

        self.dropout2 = nn.Dropout(drop_rate)

        self.use_scaled_init = use_scaled_init

        if norm_type == "rmsnorm":
            self.ffn_norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        else:
            self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, "dropout_add_ln is not installed"
            assert isinstance(self.norm1, nn.LayerNorm) and isinstance(self.dropout1, nn.Dropout)
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False
        self.deepnorm = (
            deepnorm  # Applying deepnorm under prenorm is equivalent to using the initialization of deepnorm,
        )
        # and will not affect the training process.
        if deepnorm:
            self.deepnorm_reset_parameters()
        else:
            self.reset_parameters()

    def deepnorm_reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "Wqkv" in name:
                    wq, wk, wv = param.data.chunk(3, dim=0)
                    xavier_normal_tensor_parallel(
                        wq, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=1.0
                    )
                    xavier_normal_tensor_parallel(
                        wk, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=1.0
                    )
                    xavier_normal_tensor_parallel(
                        wv, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=self.deepnorm_beta
                    )
                else:  # dense
                    xavier_normal_tensor_parallel(
                        param.data,
                        partition_dim=1,
                        tp_degree=gpc.get_world_size(ParallelMode.TENSOR),
                        gain=self.deepnorm_beta,
                    )

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1 and "bias" in name:
                    param.data.zero_()
                else:
                    if "w1" in name or "w2" in name:
                        xavier_normal_tensor_parallel(
                            param.data,
                            partition_dim=0,
                            tp_degree=gpc.get_world_size(ParallelMode.TENSOR),
                            gain=self.deepnorm_beta,
                        )
                    else:  # w3
                        xavier_normal_tensor_parallel(
                            param.data,
                            partition_dim=1,
                            tp_degree=gpc.get_world_size(ParallelMode.TENSOR),
                            gain=self.deepnorm_beta,
                        )

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "Wqkv" in name:
                    normal_(std=0.006)(param.data)
                else:
                    if self.use_scaled_init:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        normal_(std=0.0015)(param.data)

            for name, param in self.feed_forward.named_parameters():
                if param.ndim == 1 and "bias" in name:
                    param.data.zero_()
                else:
                    if self.use_swiglu:
                        if self.use_scaled_init:
                            if "w1" in name:  # or "w2" in name:
                                normal_(std=0.006)(param.data)
                            elif "w2" in name:
                                # normal_(std=0.0015)(param.data)
                                scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                            else:
                                normal_(std=0.0015)(param.data)
                        else:
                            if "w1" in name or "w2" in name:
                                normal_(std=0.006)(param.data)
                            else:
                                normal_(std=0.0015)(param.data)
                    else:
                        if self.use_scaled_init:
                            if "fc1" in name:
                                normal_(std=0.006)(param.data)
                            else:
                                scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                        else:
                            if "fc1" in name:
                                normal_(std=0.006)(param.data)
                            else:
                                normal_(std=0.0015)(param.data)

    def forward(
        self, hidden_states, residual=None, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None
    ):
        if self.checkpoint and self.training:
            return activation_checkpoint(
                self._forward, False, hidden_states, residual, cu_seqlens, indexes, inference_params, max_seqlen
            )
        else:
            return self._forward(hidden_states, residual, cu_seqlens, indexes, inference_params, max_seqlen)

    def _forward(
        self, hidden_states=None, residual=None, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """
        if self.prenorm:

            def _dropout_and_norm_attn(_residual, _hidden_states):
                _dropped = self.dropout1(_hidden_states)
                _residual = (_dropped + _residual) if _residual is not None else _dropped
                _hidden_states = self.attention_norm(_residual.to(dtype=self.attention_norm.weight.dtype))

                return _residual, _hidden_states

            if self.dropout_selective_checkpoint:
                residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, residual, hidden_states)
            else:
                residual, hidden_states = _dropout_and_norm_attn(residual, hidden_states)

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            mixer_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "indexes": indexes,
                "inference_params": inference_params,
            }
            hidden_states = self.attention(hidden_states, **mixer_kwargs)

            if not isinstance(self.feed_forward, nn.Identity):
                if not self.fused_dropout_add_ln:

                    def _dropout_and_norm_ffn(_residual, _hidden_states):
                        _dropped = self.dropout2(_hidden_states)
                        _residual = (_dropped + _residual) if _residual is not None else _dropped
                        _hidden_states = self.ffn_norm(_residual.to(torch.float32))

                        return _residual, _hidden_states

                    if self.dropout_selective_checkpoint:
                        residual, hidden_states = activation_checkpoint(
                            _dropout_and_norm_ffn, False, residual, hidden_states
                        )
                    else:
                        residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                hidden_states = self.feed_forward(hidden_states)

            return hidden_states + residual
        else:
            assert residual is None
            mixer_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "indexes": indexes,
                "inference_params": inference_params,
            }
            mixer_out = self.attention(hidden_states, **mixer_kwargs)
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            hidden_states = self.attention_norm(self.dropout1(mixer_out) + hidden_states * self.deepnorm_alpha).to(
                dtype=self.norm1.weight.dtype
            )
            if not isinstance(self.feed_forward, nn.Identity):
                mlp_out = self.feed_forward(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                hidden_states = self.ffn_norm((self.dropout2(mlp_out)) + hidden_states * self.deepnorm_alpha).to(
                    dtype=self.norm2.weight.dtype
                )
            return hidden_states


class PackedFlashPipelineConvertedLLAMA1D2(nn.Module):
    """
    The pipeline of LLAMA implement

    Args:
        num_layers (int): The number of layer. 12 by default
        hidden_size (int): The size of hidden state. 768 by default
        num_attention_heads (int): The number of attention head. 12 by default
        vocab_size (int): The size of vocabulary. 50304 by default
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default
        dtype (torch.dtype): The type of data. torch.float by default
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        checkpoint_fraction (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 1.0 by default
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
        apply_post_layer_norm (bool): Whether to use postnorm(True) or prenorm(False). False by default.
        first (bool): Whether input embedding layer or not. False by default
        last (bool): Whether output embedding layer or not. False by default
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    True by default
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default
        device (Optional[Union[str, torch.device]]): The device will be used. None by default
        no_bias (bool): Whether the bias is needed for linears. False by default
        deepnorm (bool): Whether to use deepnorm or not. False by default
        total_layers (int): The total number of layers. Will be used in deepnorm. -1 by default
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default

    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_kv_attention_heads: int = 12,
        vocab_size: int = 50304,
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        dtype: Optional[torch.dtype] = torch.float,
        checkpoint: bool = False,
        checkpoint_fraction: float = 1.0,
        layer_norm_epsilon: float = 1e-6,
        apply_post_layer_norm: bool = False,
        first: bool = False,
        last: bool = False,
        embed_split_hidden: bool = True,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        device: Optional[torch.device] = None,
        no_bias: bool = False,
        deepnorm: bool = False,
        total_layers: int = -1,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        is_reward: bool = False,
        adapt_hf: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
    ):
        super().__init__()
        if is_reward:
            head_cls = RewardModelLinear
        else:
            head_cls = ScaleColumnParallelLinear
        if checkpoint_fraction <= 0:
            checkpoint = False
        if not checkpoint:
            checkpoint_fraction = 0
        checkpoint_layer_num = num_layers * checkpoint_fraction
        self.adapt_hf = adapt_hf
        # if not embed_split_hidden:
        #     raise RuntimeError("llama split embedding layer in the hidden dimension")

        if first:
            if embed_split_hidden:
                self.tok_embeddings = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
            else:
                self.tok_embeddings = ParallelGPT2Embeddings(
                    embed_dim=hidden_size,
                    vocab_size=vocab_size,
                    max_position_embeddings=-1,
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    padding_idx=None,
                    sequence_parallel=gpc.config.model.sequence_parallel,
                    device=device,
                    dtype=dtype,
                )
            for _, param in self.tok_embeddings.named_parameters():
                normal_(std=0.0052)(param)
        self.embed_grad_scale = embed_grad_scale
        self.layers = nn.ModuleList(
            [
                PackedFlashConvertedLLAMALayer1D2(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_attention_heads=num_kv_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    apply_post_layer_norm=apply_post_layer_norm,
                    fused_dropout_add_ln=False,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    no_bias=no_bias,
                    deepnorm=deepnorm,
                    total_layers=total_layers,  # For deepnorm
                    norm_type=norm_type,
                    adapt_hf=adapt_hf,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            if not apply_post_layer_norm:
                if norm_type == "rmsnorm":
                    RMSNorm = try_import_RMSNorm()
                    self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
                else:
                    self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

            self.output = head_cls(
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias=False,
                device=device,
                dtype=dtype,
                weight_scale=embed_grad_scale,
            )

            for _, param in self.output.named_parameters():
                normal_(std=0.0052)(param)
        self.parallel_output = parallel_output
        self.apply_post_layer_norm = apply_post_layer_norm

        # need to assign tp attribute so that colossalai know it is tensor parallel module
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["tok_embeddings", "output"]:
                if hasattr(self, name):
                    for param in getattr(self, name).parameters():
                        setattr(param, IS_TENSOR_PARALLEL, True)

    def forward(self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "tok_embeddings"):
            hidden_states = self.tok_embeddings(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )
        if isinstance(cu_seqlens, list):
            # assert len(cu_seqlens) == 1
            cu_seqlens = cu_seqlens[0].to(hidden_states.device)
            # the batch dimension with a size of 1 should be directly squeezed off.
        if cu_seqlens is not None:
            hidden_states = hidden_states.squeeze(0)  # If cu_seqlens is passed in，it indicated a packed state，
            cu_seqlens = cu_seqlens.squeeze(0)

        if indexes is not None:
            assert len(indexes) == 1
            indexes = indexes[
                0
            ]  # The indexes are used to indicate the actual position IDs of each token in the packed input.
        max_seqlen = None
        if cu_seqlens is not None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        for _, block in enumerate(self.layers):
            hidden_states = block(
                hidden_states,
                residual=None,
                cu_seqlens=cu_seqlens,
                indexes=indexes,
                inference_params=inference_params,
                max_seqlen=max_seqlen,
            )
        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)
        if hasattr(self, "output"):
            hidden_states = self.output(hidden_states)
        if not self.parallel_output:
            hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=-1)

        return hidden_states


def _build_generic_converted_llama_pipeline_1d2(num_layers, num_chunks, device=torch.device("cuda"), **kwargs):
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0

    all_parts = partition_uniform_with_embed(num_layers, pipeline_size, num_chunks)
    if gpc.is_initialized(ParallelMode.GLOBAL):
        if gpc.get_global_rank() == 0:
            logger.info(f"The layer sharding is {all_parts}.")
    parts = all_parts[pipeline_rank]

    if kwargs["checkpoint"] is False:
        # if pipeline_rank <= 4:
        #     kwargs["checkpoint"] = True
        #     kwargs["checkpoint_fraction"] = 0.4
        # else:
        #     kwargs["checkpoint"] = True
        #     kwargs["checkpoint_fraction"] = 0.01
        pass
    else:
        kwargs["checkpoint_fraction"] = 1.0

    models = []
    kwargs["total_layers"] = num_layers
    start, end = None, None
    for start, end in parts:
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        kwargs["last"] = (
            end == num_layers and len(all_parts[-1]) != 0
        )  # If there is no content in the final layer, assign the last layer.
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start

        chunk = PackedFlashPipelineConvertedLLAMA1D2(
            **filter_kwargs(PackedFlashPipelineConvertedLLAMA1D2.__init__, kwargs)
        ).to(device)

        setattr(chunk, "first_layer", start)
        setattr(chunk, "last_layer", end)

        models.append(chunk)

    torch.distributed.barrier()

    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model


@MODEL_INITIALIZER.register_module(module_name=MODEL_TYPE)
def Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D2(
    num_chunks=1,
    checkpoint=False,
    dtype=torch.float,
    embed_split_hidden=False,
    num_layers=48,
    hidden_size=2048,
    vocab_size=50304,
    embed_grad_scale=0.1,
    parallel_output=True,
    num_attention_heads=32,
    num_kv_attention_heads=None,
    mlp_ratio=4.0,
    apply_post_layer_norm=False,
    no_bias=False,
    deepnorm=False,
    residual_in_fp32=False,
    norm_type="rmsnorm",
    drop_rate=0,
    attn_drop_rate=0,
    model_type="llama",
    layer_norm_epsilon=1e-6,
    is_reward=False,
    adapt_hf=False,
    dropout_selective_checkpoint=True,
    use_scaled_init: bool = True,
    use_swiglu: bool = True,
    use_flash_attn: bool = True,
    sequence_parallel: bool = False,
):
    assert model_type == "llama", f"Only support llama for this initilization, not `{model_type}`"

    del use_flash_attn, sequence_parallel

    # residual_in_fp32 cannot be used temporarily because this parameter requires inconsistent data types to
    # be passed between pipelines, which requires significant modifications to colossalai.
    cfg = dict(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_attention_heads=num_kv_attention_heads if num_kv_attention_heads else num_attention_heads,
        checkpoint=checkpoint,
        dtype=dtype,
        embed_split_hidden=embed_split_hidden,
        vocab_size=vocab_size,
        embed_grad_scale=embed_grad_scale,
        parallel_output=parallel_output,
        mlp_ratio=mlp_ratio,
        apply_post_layer_norm=apply_post_layer_norm,
        no_bias=no_bias,
        deepnorm=deepnorm,
        residual_in_fp32=residual_in_fp32,
        norm_type=norm_type,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        is_reward=is_reward,
        adapt_hf=adapt_hf,
        dropout_selective_checkpoint=dropout_selective_checkpoint,
        use_scaled_init=use_scaled_init,
        use_swiglu=use_swiglu,
    )
    return _build_generic_converted_llama_pipeline_1d2(num_layers=num_layers, num_chunks=num_chunks, **cfg)
