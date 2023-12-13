# Copyright (c) InternLM. All rights reserved.
import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from internlm.core.context import IS_SEQUENCE_PARALLEL, IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.initialize.initialize_tensor import (
    normal_,
    scaled_init_method_normal,
    scaled_init_method_uniform,
    uniform_,
)
from internlm.model.embedding import Embedding1D, RotaryEmbedding
from internlm.model.linear import (
    ColumnParallelLinearTorch,
    FeedForward,
    RewardModelLinear,
    RowParallelLinearTorch,
    ScaleColumnParallelLinear,
)
from internlm.model.utils import gather_forward_split_backward, try_import_RMSNorm
from internlm.solver.pipeline_utils import partition_uniform
from internlm.utils.checkpoint import activation_checkpoint
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER

try:
    from flash_attn import flash_attn_varlen_kvpacked_func
    from flash_attn.flash_attn_interface import FlashAttnVarlenKVPackedFunc
    from flash_attn.modules.embedding import ParallelGPT2Embeddings
    from flash_attn.modules.mha import (
        CrossAttention,
        FlashCrossAttention,
        FlashSelfAttention,
        SelfAttention,
        _update_kv_cache,
    )
    from flash_attn.modules.mlp import ParallelFusedMLP
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    pass

MODEL_TYPE = "LLAMA2"

logger = get_logger(__file__)
RMSNorm = try_import_RMSNorm()


class MHA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of kv attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        bias (boolean): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                        output projection. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        use_flash_attn (boolean): Whether to use flash attention or not.If False, vanilla attention module will be used.
                                    True by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        rot_embed_HF_impl: rotary embedding hf implementation. False by default.


    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        rope_base: int = 10000,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        use_flash_attn: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        rot_embed_HF_impl: Optional[bool] = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        self.head_dim = self.embed_dim // num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.dtype = dtype

        self.rot_embed_HF_impl = rot_embed_HF_impl
        sequence_parallel = gpc.config.parallel.get("sequence_parallel", False)

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim, base=rope_base, scale_base=rotary_emb_scale_base, device=device
            )

        # notice here should change bias=True
        self.wq = ColumnParallelLinearTorch(
            embed_dim,
            embed_dim,
            process_group,
            bias=bias,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.wk = ColumnParallelLinearTorch(
            embed_dim,
            self.kv_dim,
            process_group,
            bias=bias,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.wv = ColumnParallelLinearTorch(
            embed_dim,
            self.kv_dim,
            process_group,
            bias=bias,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )

        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )

        self.inner_cross_attn_causal = causal
        self.inner_cross_attn_softmax_scale = softmax_scale
        self.inner_cross_attn_dropout = dropout

        # output projection always have the bias (for now)
        self.wo = RowParallelLinearTorch(
            embed_dim,
            embed_dim,
            process_group,
            bias=bias,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        # need to assign tp attribute so that internlm know it is tensor parallel module
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["wo", "wq", "wk", "wv"]:
                for param in getattr(self, name).parameters():
                    setattr(param, IS_TENSOR_PARALLEL, True)

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get("indexes", None) is not None:
            return self._packed_forward(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._forward(x=x, seqlen=seqlen, inference_params=inference_params, **kwargs)

    def _forward(self, x, seqlen=None, inference_params=None, **kwargs):  # pylint: disable=W0613
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        bsz, _, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        if seqlen is None:
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)
        else:
            q = rearrange(q, "(b s) (h d) -> b s h d", s=seqlen, d=self.head_dim)
            k = rearrange(k, "(b s) (h d) -> b s h d", s=seqlen, d=self.head_dim)
            v = rearrange(v, "(b s) (h d) -> b s h d", s=seqlen, d=self.head_dim)

        if not self.rot_embed_HF_impl:
            q = torch.cat([q[..., ::2], q[..., 1::2]], dim=-1)
            k = torch.cat([k[..., ::2], k[..., 1::2]], dim=-1)
        if inference_params is None:
            if self.rotary_emb_dim > 0:
                q = self.rotary_emb._single_eval_forward(q)
                k = self.rotary_emb._single_eval_forward(k)
            kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
            if self.dtype is torch.float32 and self.use_flash_attn:
                if q.dtype not in [torch.float16, torch.bfloat16]:
                    q = q.to(torch.bfloat16)
                if kv.dtype not in [torch.float16, torch.bfloat16]:
                    kv = kv.to(torch.bfloat16)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    context = self.inner_cross_attn(q, kv).to(self.dtype)
            else:
                context = self.inner_cross_attn(q, kv)

        else:
            assert self.rotary_emb_dim > 0
            if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
                empties = inference_params.attention_mask[..., -1].sum(dim=-1)
                moved_q = q.clone()
                moved_k = k.clone()
                if inference_params.sequence_len_offset == 0:
                    for i in range(len(empties)):
                        if empties[i] != 0:
                            moved_q[i][: -empties[i]] = q[i][empties[i] :]
                            moved_k[i][: -empties[i]] = k[i][empties[i] :]
                    moved_q = self.rotary_emb._single_eval_forward(
                        moved_q, seqlen_offset=inference_params.sequence_len_offset
                    )
                    moved_k = self.rotary_emb._single_eval_forward(
                        moved_k, seqlen_offset=inference_params.sequence_len_offset
                    )
                    for i in range(len(empties)):
                        if empties[i] != 0:
                            q[i][empties[i] :] = moved_q[i][: -empties[i]]
                            k[i][empties[i] :] = moved_k[i][: -empties[i]]
                        else:
                            q[i] = moved_q[i]
                            k[i] = moved_k[i]
                else:
                    q = q.squeeze(1)
                    k = k.squeeze(1)
                    q = self.rotary_emb._single_forward(
                        q,
                        inference_params.sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device)
                        - empties,
                    ).unsqueeze(1)
                    k = self.rotary_emb._single_forward(
                        k,
                        inference_params.sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device)
                        - empties,
                    ).unsqueeze(1)
            else:
                raise NotImplementedError(
                    "You should make sure you are aware that you are changing the method of generating."
                    "According to your generation function instead of inference/seq_generator_module.py, "
                    "You may implement here for normal running."
                )

            kv = torch.stack([k, v], dim=2)

            assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
            if hasattr(inference_params, "window_size") and inference_params.window_size is not None:
                if inference_params.window_size <= inference_params.sequence_len_offset:
                    assert kv.size(1) == 1, "update kv lenth more than 1"
                    inference_params.key_value_memory_dict[self.layer_idx][
                        :, inference_params.keep_first : inference_params.window_size - 1, ...
                    ] = inference_params.key_value_memory_dict[self.layer_idx][
                        :, -(inference_params.window_size - 1 - inference_params.keep_first) :, ...
                    ].clone()
                    inference_params.real_sequence_len_offset = inference_params.sequence_len_offset
                    inference_params.sequence_len_offset = inference_params.window_size - 1

                    kv = _update_kv_cache(kv, inference_params, self.layer_idx)

                    inference_params.sequence_len_offset = inference_params.real_sequence_len_offset
                else:
                    kv = _update_kv_cache(kv, inference_params, self.layer_idx)
            else:
                kv = _update_kv_cache(kv, inference_params, self.layer_idx)

            # When using FP16, there is a high probability of NAN in the KV.
            # Since NAN cannot be removed by multiplying with and 0, it needs
            # to be removed manually here.
            kv = torch.where(torch.isnan(kv), 0, kv)

            if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
                assert self.use_flash_attn is True
                if inference_params.sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
                    attn_mask = inference_params.attention_mask[:, None, ...]
                    attn_mask = torch.logical_or(
                        torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask
                    )
                    attn_mask4flsh = ~attn_mask[:, :, -1, :].view(bsz, -1)
                    cu_seqlens = torch.concat(
                        [
                            torch.tensor([0], dtype=torch.int32, device=attn_mask4flsh.device),
                            attn_mask4flsh.sum(dim=-1).to(dtype=torch.int32),
                        ],
                        dim=0,
                    )
                    cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                    max_seqlen_q = attn_mask4flsh.shape[-1]
                    max_seqlen_k = attn_mask4flsh.shape[-1]
                    total_q = q.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1])
                    total_kv = kv.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1, 1)).view(
                        -1, kv.shape[-3], kv.shape[-2], kv.shape[-1]
                    )
                    if self.dtype is torch.float32:
                        if total_q.dtype not in [torch.float16, torch.bfloat16]:
                            total_q = total_q.to(torch.bfloat16)
                        if total_kv.dtype not in [torch.float16, torch.bfloat16]:
                            total_kv = total_kv.to(torch.bfloat16)
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            output = FlashAttnVarlenKVPackedFunc.apply(
                                total_q,
                                total_kv,
                                cu_seqlens,
                                cu_seqlens,
                                max_seqlen_q,
                                max_seqlen_k,
                                0.0,
                                None,
                                True,
                                False,
                            ).to(self.dtype)
                    else:
                        output = FlashAttnVarlenKVPackedFunc.apply(
                            total_q,
                            total_kv,
                            cu_seqlens,
                            cu_seqlens,
                            max_seqlen_q,
                            max_seqlen_k,
                            0.0,
                            None,
                            True,
                            False,
                        )

                    context = torch.zeros_like(q)
                    context = context.masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)

                else:
                    attn_mask = inference_params.attention_mask[:, -1, :].view(bsz, 1, 1, -1)
                    if hasattr(inference_params, "window_size") and inference_params.window_size is not None:
                        if inference_params.window_size <= inference_params.sequence_len_offset:
                            attn_mask = torch.concat(
                                [
                                    attn_mask[..., : inference_params.keep_first],
                                    attn_mask[..., -(inference_params.window_size - inference_params.keep_first) :],
                                ],
                                dim=-1,
                            )

                    k, v = torch.chunk(kv, 2, dim=2)
                    k = k.squeeze(2)
                    v = v.squeeze(2)
                    sp = k.shape
                    expansion = q.size(2) // k.size(2)
                    scores = torch.einsum(
                        "blhd,bnhd->bhln",
                        q,
                        k.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                    ) / math.sqrt(q.size(-1))
                    scores = scores.masked_fill(attn_mask, -65000.0)
                    scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                    context = torch.einsum(
                        "bhmn,bnhd->bmhd",
                        scores,
                        v.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                    )
            else:
                if self.dtype is torch.float32 and self.use_flash_attn:
                    if q.dtype not in [torch.float16, torch.bfloat16]:
                        q = q.to(torch.bfloat16)
                    if kv.dtype not in [torch.float16, torch.bfloat16]:
                        kv = kv.to(torch.bfloat16)
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        context = self.inner_cross_attn(q, kv, causal=True).to(self.dtype)
                else:
                    context = self.inner_cross_attn(q, kv, causal=True)
        if seqlen is None:
            context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")
        out = self.wo(context)
        return out

    def _packed_forward(self, x, inference_params=None, **kwargs):
        """
        we delete seqlen=None for lint check, cause this arg is not used.

        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        assert self.use_flash_attn is True
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = rearrange(q, "t (h d) -> t h d", d=self.head_dim)
        k = rearrange(k, "t (h d) -> t h d", d=self.head_dim)
        v = rearrange(v, "t (h d) -> t h d", d=self.head_dim)

        # qkv shift
        # the rotary embedding in flash attention module in performed by separating the front and back parts, while
        # most of others are done by odd-even methods.
        if not self.rot_embed_HF_impl:
            q = torch.cat([q[..., ::2], q[..., 1::2]], dim=-1)
            k = torch.cat([k[..., ::2], k[..., 1::2]], dim=-1)

        indexes = kwargs.pop("indexes")
        q = self.rotary_emb._single_forward(q, indexes=indexes)
        k = self.rotary_emb._single_forward(k, indexes=indexes)

        if inference_params is None:
            kv = torch.concat([k.unsqueeze(1), v.unsqueeze(1)], dim=1)
            if self.dtype is torch.float32:
                if q.dtype not in [torch.float16, torch.bfloat16]:
                    q = q.to(torch.bfloat16)
                if kv.dtype not in [torch.float16, torch.bfloat16]:
                    kv = kv.to(torch.bfloat16)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    context = flash_attn_varlen_kvpacked_func(
                        q=q,
                        kv=kv,
                        cu_seqlens_q=kwargs["cu_seqlens"],
                        cu_seqlens_k=kwargs["cu_seqlens"],
                        max_seqlen_q=kwargs["max_seqlen"],
                        max_seqlen_k=kwargs["max_seqlen"],
                        dropout_p=self.inner_cross_attn_dropout,
                        softmax_scale=self.inner_cross_attn_softmax_scale,
                        causal=self.inner_cross_attn_causal,
                    ).to(self.dtype)
            else:
                context = flash_attn_varlen_kvpacked_func(
                    q=q,
                    kv=kv,
                    cu_seqlens_q=kwargs["cu_seqlens"],
                    cu_seqlens_k=kwargs["cu_seqlens"],
                    max_seqlen_q=kwargs["max_seqlen"],
                    max_seqlen_k=kwargs["max_seqlen"],
                    dropout_p=self.inner_cross_attn_dropout,
                    softmax_scale=self.inner_cross_attn_softmax_scale,
                    causal=self.inner_cross_attn_causal,
                )
        else:
            raise RuntimeError("Not support this right now")

        context = rearrange(context, "b h d -> b (h d)")  # recover shape
        out = self.wo(context)
        return out


class PackedFlashLlamaLayer1D(nn.Module):
    """
    1D Packed Flash Llama Layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        num_kv_attention_heads (int): The number of kv attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        apply_post_layer_norm (bool): Whether use post layer norm. False by default.
        fused_dropout_add_ln (bool): Whether use fused dropout add ln. True by default.
        no_bias (bool): Whether remove bias. False by default.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
        adapt_hf (bool): Whether adapt hf. False by default.
        dropout_selective_checkpoint (bool): Whether use dropout selective checkpoint. True by default.
        use_scaled_init (bool): Whether use scaled init. True by default.
        use_swiglu (bool): Whether use swiglu. True by default.
        use_flash_attn (bool): Whether use flash-attn. True by default.
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
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
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        layer_idx: int = 0,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        apply_post_layer_norm: bool = False,
        fused_dropout_add_ln: bool = True,
        no_bias: bool = False,
        norm_type: str = "rmsnorm",
        adapt_hf: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx
        self.use_flash_attn = use_flash_attn
        self.prenorm = not apply_post_layer_norm
        assert not fused_dropout_add_ln, "dropout_add_layer_norm can not be used here"
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.attn_wqkv_init_std = attn_wqkv_init_std
        self.attn_other_init_std = attn_other_init_std
        self.ffn_uplayer_init_std = ffn_uplayer_init_std
        self.ffn_other_init_std = ffn_other_init_std

        head_dim = hidden_size // num_attention_heads
        self.attention = MHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_attention_heads,
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
            rot_embed_HF_impl=adapt_hf,
            bias=not no_bias,
            rope_base=rope_base,
        )

        self.dropout1 = nn.Dropout(drop_rate)
        if norm_type == "rmsnorm":
            self.attention_norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
            self.ffn_norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        else:
            self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, "dropout_add_ln is not installed"
            assert isinstance(self.attention_norm, nn.LayerNorm) and isinstance(self.dropout1, nn.Dropout)

        sequence_parallel = gpc.config.parallel.get("sequence_parallel", False)
        if use_swiglu:
            self.feed_forward = FeedForward(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias=False,
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
                bias1=False,
                bias2=False,
                sequence_parallel=sequence_parallel,
                checkpoint_lvl=0,
                heuristic="auto",
                device=device,
                dtype=dtype,
            )

        for _, param in self.feed_forward.named_parameters():
            if gpc.get_world_size(ParallelMode.TENSOR) > 1:
                setattr(param, IS_TENSOR_PARALLEL, True)
        for param in self.attention_norm.parameters():
            if gpc.config.parallel.sequence_parallel is True:
                setattr(param, IS_SEQUENCE_PARALLEL, True)
        for param in self.ffn_norm.parameters():
            if gpc.config.parallel.sequence_parallel is True:
                setattr(param, IS_SEQUENCE_PARALLEL, True)

        self.dropout2 = nn.Dropout(drop_rate)
        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False

        if init_type == "normal":
            self.init_func = normal_
            self.scaled_init_func = scaled_init_method_normal
        else:
            self.init_func = uniform_
            self.scaled_init_func = scaled_init_method_uniform

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wq" in name or "wk" in name or "wv" in name:
                    self.init_func(std=self.attn_wqkv_init_std)(param.data)
                elif self.use_scaled_init:  # wo
                    self.scaled_init_func(sigma=self.attn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                else:
                    self.init_func(std=self.attn_other_init_std)(param.data)

            for name, param in self.feed_forward.named_parameters():
                if self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        self.init_func(
                            std=self.ffn_uplayer_init_std if "w1" in name or "w3" in name else self.ffn_other_init_std
                        )(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        self.init_func(std=self.ffn_uplayer_init_std if "fc1" in name else self.ffn_other_init_std)(
                            param.data
                        )

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
            residual: hidden_states = Attn/MLP(LN(residual))
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
            hidden_states = self.attention_norm(self.dropout1(mixer_out) + hidden_states).to(
                dtype=self.attention_norm.weight.dtype
            )
            if not isinstance(self.feed_forward, nn.Identity):
                mlp_out = self.feed_forward(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                hidden_states = self.ffn_norm((self.dropout2(mlp_out)) + hidden_states).to(
                    dtype=self.ffn_norm.weight.dtype
                )
            return hidden_states


class PackedFlashLlama1D(nn.Module):
    """
    1D Packed Flash Llama.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        num_attention_heads (int): The number of attention head. 12 by default.
        num_kv_attention_heads (int): The number of kv attention head. 12 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        checkpoint_fraction (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 1.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    True by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        apply_post_layer_norm (bool): Whether use post layer norm. False by default.
        no_bias (bool): Whether remove bias. False by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        adapt_hf (bool): Whether adapt hf. False by default.
        is_reward (bool): Whether use is_reward. False by default.
        dropout_selective_checkpoint (bool): Whether dropout selective checkpoint. True by default.
        use_scaled_init (bool): Whether use scaled init. True by default.
        use_swiglu (bool): Whether use swiglu. True by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
        embedding_init_std (float): std used to init embedding weight. 0.02 by default,
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        out_head_init_std (float): std used to init output lmhead weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_kv_attention_heads: int = 12,
        vocab_size: int = 50304,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        checkpoint: bool = False,
        checkpoint_fraction: float = 1.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_split_hidden: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        device: Optional[torch.device] = None,
        apply_post_layer_norm=False,
        no_bias=False,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        adapt_hf: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
        embedding_init_std: float = 0.02,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        out_head_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
    ):
        super().__init__()

        self.use_flash_attn = use_flash_attn
        if checkpoint_fraction <= 0:
            checkpoint = False
        if not checkpoint:
            checkpoint_fraction = 0
        checkpoint_layer_num = num_layers * checkpoint_fraction
        sequence_parallel = gpc.config.parallel.get("sequence_parallel", False)
        if is_reward:
            head_cls = RewardModelLinear
        else:
            head_cls = ScaleColumnParallelLinear
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
                    sequence_parallel=sequence_parallel,
                    device=device,
                    dtype=dtype,
                )
            for _, param in self.tok_embeddings.named_parameters():
                if init_type == "normal":
                    normal_(std=embedding_init_std)(param)
                else:
                    uniform_(std=embedding_init_std)(param)
                if gpc.get_world_size(ParallelMode.TENSOR) > 1:
                    setattr(param, IS_TENSOR_PARALLEL, True)
        self.embed_grad_scale = embed_grad_scale
        self.layers = nn.ModuleList(
            [
                PackedFlashLlamaLayer1D(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_attention_heads=num_kv_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    apply_post_layer_norm=apply_post_layer_norm,
                    fused_dropout_add_ln=False,
                    no_bias=no_bias,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    use_flash_attn=use_flash_attn,
                    adapt_hf=adapt_hf,
                    attn_wqkv_init_std=attn_wqkv_init_std,
                    attn_other_init_std=attn_other_init_std,
                    ffn_uplayer_init_std=ffn_uplayer_init_std,
                    ffn_other_init_std=ffn_other_init_std,
                    init_type=init_type,
                    rope_base=rope_base,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            if not apply_post_layer_norm:
                if norm_type == "rmsnorm":
                    self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
                else:
                    self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
                for param in self.norm.parameters():
                    if gpc.config.parallel.sequence_parallel is True:
                        setattr(param, IS_SEQUENCE_PARALLEL, True)

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
                if init_type == "normal":
                    normal_(std=out_head_init_std)(param)
                else:
                    uniform_(std=out_head_init_std)(param)
                if gpc.get_world_size(ParallelMode.TENSOR) > 1:
                    setattr(param, IS_TENSOR_PARALLEL, True)

        self.parallel_output = parallel_output

    def forward(self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "tok_embeddings"):
            hidden_states = self.tok_embeddings(input_ids)
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
            hidden_states = self.norm(hidden_states.float())

        extra_hidden_states_list = None
        if hasattr(self, "output"):
            hidden_states = self.output(hidden_states)

        if not self.parallel_output:
            hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=-1)
            if extra_hidden_states_list is not None:
                extra_hidden_states_list = [
                    gather_forward_split_backward(extra_hidden_states, ParallelMode.TENSOR, dim=-1)
                    for extra_hidden_states in extra_hidden_states_list
                ]

        if extra_hidden_states_list is not None:
            return (hidden_states, extra_hidden_states_list)

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
    kwargs["checkpoint_fraction"] = 1.0
    start_idx, end_idx = 0, 0
    for start, end in parts:
        start_idx, end_idx = start, end
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        # If there is no content in the final layer, assign the last layer.
        kwargs["last"] = end == num_layers and len(all_parts[-1]) != 0
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start
        chunk = PackedFlashLlama1D(**filter_kwargs(PackedFlashLlama1D.__init__, kwargs)).to(device)

        models.append(chunk)
    torch.distributed.barrier()
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    setattr(model, "first_layer", start_idx)
    setattr(model, "last_layer", end_idx)
    return model


@MODEL_INITIALIZER.register_module(module_name=MODEL_TYPE)
def build_model_with_cfg(
    num_chunks=1,
    checkpoint=False,
    dtype=torch.float,
    embed_split_hidden=False,
    num_layers=48,
    hidden_size=2048,
    vocab_size=50304,
    embed_grad_scale=1,
    parallel_output=True,
    num_attention_heads=32,
    num_kv_attention_heads=None,
    mlp_ratio=4.0,
    residual_in_fp32=False,
    norm_type="rmsnorm",
    adapt_hf=False,
    drop_rate=0,
    attn_drop_rate=0,
    apply_post_layer_norm=False,  # pylint: disable=W0613
    no_bias=False,
    deepnorm=False,
    layer_norm_epsilon=1e-5,
    is_reward=False,
    dropout_selective_checkpoint=True,
    use_scaled_init: bool = True,
    use_swiglu: bool = True,
    use_flash_attn: bool = True,
    embedding_init_std: float = 0.02,
    attn_wqkv_init_std: float = 0.02,
    attn_other_init_std: float = 0.02,
    ffn_uplayer_init_std: float = 0.02,
    ffn_other_init_std: float = 0.02,
    out_head_init_std: float = 0.02,
    init_type: str = "normal",
    rope_base: int = 10000,
):
    """
    Builde model with config

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
        num_kv_attention_heads (int): The number of kv attention head. None by default.
        mlp_ratio (int): The ratio of MLP layers. 4.0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default. It cannot be used temporarily
                                 because this parameter requires inconsistent data types to be passed between pipelines,
                                 which requires significant modifications to internlm.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        adapt_hf (bool): Whether adapt hf. False by default.
        drop_rate (float): The dropout rate of input hidden state. 0 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        apply_post_layer_norm (bool): Whether to apply post layer norm. False by default.
        no_bias (bool): Whether remove bias. False by default.
        deepnorm (bool): Whether us deepnorm. False by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        is_reward (bool): Whether to use reward model. False by default.
        dropout_selective_checkpoint (bool): It can only be enabled when checkpoint is disabled. True by default.
        use_scaled_init (bool): Whether to use scaled init. True by default.
        use_swiglu (bool): Whether to use swiglu. True by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
        embedding_init_std (float): std used to init embedding weight. 0.02 by default,
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        out_head_init_std (float): std used to init output lmhead weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
    """
    if deepnorm:
        raise AssertionError("deepnorm will not be supported in future versions." "Use early versions if necessary.")

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
        residual_in_fp32=residual_in_fp32,
        norm_type=norm_type,
        adapt_hf=adapt_hf,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        is_reward=is_reward,
        dropout_selective_checkpoint=dropout_selective_checkpoint,
        use_scaled_init=use_scaled_init,
        use_swiglu=use_swiglu,
        use_flash_attn=use_flash_attn,
        embedding_init_std=embedding_init_std,
        attn_wqkv_init_std=attn_wqkv_init_std,
        attn_other_init_std=attn_other_init_std,
        ffn_uplayer_init_std=ffn_uplayer_init_std,
        ffn_other_init_std=ffn_other_init_std,
        out_head_init_std=out_head_init_std,
        init_type=init_type,
        rope_base=rope_base,
    )

    return _build_generic_model_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)
