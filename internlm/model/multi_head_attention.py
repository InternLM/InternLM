#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
from einops import rearrange
from flash_attn.modules.mha import (
    CrossAttention,
    FlashCrossAttention,
    FlashSelfAttention,
    SelfAttention,
    _update_kv_cache,
)
from torch import nn

from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.embedding import RotaryEmbedding
from internlm.model.linear import ColumnParallelLinearTorch, RowParallelLinearTorch


class MHA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        bias (boolean): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                        output projection. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        use_flash_attn (boolean): Whether to use flash attention or not.If False, vanilla attention module will be used.
                                    False by default.
        sequence_parallel (boolean): If True, we're doing Tensor Parallel with sequence parallelism. An all_gather_raw
                                    of x will be done before doing the matmul.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        use_flash_attn (bool): Whether to use flash-attn. True by default.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        use_flash_attn: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device)

        # notice here should change bias=True
        self.Wqkv = ColumnParallelLinearTorch(
            embed_dim,
            3 * embed_dim,
            process_group,
            bias=True,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            **factory_kwargs,
        )  # according to https://spaces.ac.cn/archives/9577

        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )

        # output projection always have the bias (for now)
        self.out_proj = RowParallelLinearTorch(
            embed_dim,
            embed_dim,
            process_group,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            **factory_kwargs,
        )
        # need to assign tp attribute so that internlm know it is tensor parallel module
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["out_proj", "Wqkv"]:
                for param in getattr(self, name).parameters():
                    setattr(param, IS_TENSOR_PARALLEL, True)

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get("indexes", None) is not None:
            return self._packed_forward(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._forward(x=x, seqlen=seqlen, inference_params=inference_params, **kwargs)

    def _forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
        else:
            qkv = rearrange(qkv, "(b s) (three h d) -> b s three h d", s=seqlen, three=3, d=self.head_dim)

        if self.rotary_emb_dim > 0:
            kwargs["inference_params"] = inference_params
            qkv = self.rotary_emb(qkv, **kwargs)

        if inference_params is None:
            if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if qkv.dtype not in [torch.float16, torch.bfloat16]:
                        qkv = qkv.to(torch.bfloat16)
                    context = self.inner_attn(qkv).to(x.dtype)
            else:
                context = self.inner_attn(qkv)
        else:
            q = qkv[:, :, 0]
            assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
            kv = _update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
            # If we're processing the prompt, causal=None (use self.causal).
            # If we're decoding, then causal=False.
            causal = None if inference_params.sequence_len_offset == 0 else False
            context = self.inner_cross_attn(q, kv, causal=causal)

        if seqlen is None:
            context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")

        out = self.out_proj(context)
        return out

    def _packed_forward(self, x, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)  # total x hsz'
        qkv = rearrange(qkv, "t (three h d) -> t three h d", three=3, d=self.head_dim)  # total x 3 x n_head x d
        qkv = self.rotary_emb(qkv, **kwargs)
        kwargs.pop("indexes")

        if inference_params is None:
            if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if qkv.dtype not in [torch.float16, torch.bfloat16]:
                        qkv = qkv.to(torch.bfloat16)
                    context = self.inner_attn(qkv, **kwargs).to(x.dtype)
            else:
                context = self.inner_attn(qkv, **kwargs)

        else:
            raise RuntimeError("Not support this right now")

        context = rearrange(context, "b h d -> b (h d)")  # recover the shape
        out = self.out_proj(context)
        return out
