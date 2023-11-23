#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_unpadded_kvpacked_func as flash_attn_unpadded_func,
        )
    except ImportError:
        try:
            from flash_attn.flash_attn_interface import (
                flash_attn_varlen_kvpacked_func as flash_attn_unpadded_func,
            )
        except ImportError:
            raise ImportError("Please check your flash_attn version >= 1.0.5.")

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
from internlm.model.embedding import DynamicNTKScalingRotaryEmbedding, RotaryEmbedding
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
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        use_flash_attn: bool = True,
        rope_base: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            if self.use_dynamic_ntk_rope:
                self.rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    self.rotary_emb_dim,
                    base=rope_base,
                    scale_base=rotary_emb_scale_base,
                    device=device,
                    max_position_embeddings=max_position_embeddings,
                    scaling_factor=1.0,  # Currently do not support dynamic scaling.
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.rotary_emb_dim, base=rope_base, scale_base=rotary_emb_scale_base, device=device
                )

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

    def _forward(self, x, seqlen=None, inference_params=None, **kwargs):  # pylint: disable=W0613
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        bsz, _, _ = x.shape
        qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
        else:
            qkv = rearrange(qkv, "(b s) (three h d) -> b s three h d", s=seqlen, three=3, d=self.head_dim)

        if inference_params is None:
            kwargs["inference_params"] = inference_params
            qkv = self.rotary_emb(qkv, **kwargs)
            if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if qkv.dtype not in [torch.float16, torch.bfloat16]:
                        qkv = qkv.to(torch.bfloat16)
                    context = self.inner_attn(qkv).to(x.dtype)
            else:
                context = self.inner_attn(qkv)

        else:
            if self.use_dynamic_ntk_rope:
                q = qkv[:, :, 0]
                assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
                kv = _update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
                if inference_params.sequence_len_offset != 0:
                    # q shape: [bsz, 1, nheads, head_dim]
                    # kv shape: [bsz, seqlen, 2, nheads, head_dim]
                    bsz, seq_len, _, nheads, head_dim = kv.shape
                    q = torch.cat([q.new_zeros(size=(bsz, seq_len - 1, nheads, head_dim)), q], dim=1).unsqueeze(2)
                    qkv = torch.cat([q, kv], dim=2)
                    if self.rotary_emb_dim > 0:
                        qkv = self.rotary_emb(qkv)
                    q = qkv[:, [-1], 0]
                    kv = qkv[:, :, 1:]
                else:
                    if inference_params.sequence_len_offset > self.max_position_embeddings:
                        warnings.warn(
                            "Notice your prompt's length is longer than model's max_position_embeddings: "
                            f"{self.max_position_embeddings}, which will cause deviations in dynamic ntk calculations."
                        )
                    if self.rotary_emb_dim > 0:
                        kwargs["inference_params"] = inference_params
                        qkv = self.rotary_emb(qkv, **kwargs)
                        q = qkv[:, :, 0]
                        kv = qkv[:, :, 1:]
            else:
                assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
                q, k, v = (x.squeeze(2) for x in qkv.chunk(chunks=3, dim=2))
                kv = torch.stack([k, v], dim=2)
                assert self.rotary_emb_dim > 0, "You should use rotary_emb."

                if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
                    empties = inference_params.attention_mask[..., -1].sum(dim=-1)
                    if inference_params.sequence_len_offset == 0:
                        moved_q = q.clone()
                        moved_k = k.clone()
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                moved_q[i][: -empties[i]] = q[i][empties[i] :]
                                moved_k[i][: -empties[i]] = k[i][empties[i] :]
                        moved_q = self.rotary_emb._single_eval_forward(moved_q, seqlen_offset=0)
                        moved_k = self.rotary_emb._single_eval_forward(moved_k, seqlen_offset=0)
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                q[i][empties[i] :] = moved_q[i][: -empties[i]]
                                k[i][empties[i] :] = moved_k[i][: -empties[i]]
                            else:
                                q[i] = moved_q[i]
                                k[i] = moved_k[i]
                    elif not self.use_dynamic_ntk_rope:
                        if inference_params.sequence_len_offset > self.max_position_embeddings:
                            warnings.warn(
                                "Notice your prompt's length is longer than model's max_position_embeddings: "
                                f"{self.max_position_embeddings}, may cause deviations in dynamic ntk calculations."
                            )
                        q = q.squeeze(1)
                        k = k.squeeze(1)
                        q = self.rotary_emb._single_forward(
                            q,
                            inference_params.sequence_len_offset
                            * torch.ones(q.size(0), dtype=torch.int, device=q.device)
                            - empties,
                        ).unsqueeze(1)
                        k = self.rotary_emb._single_forward(
                            k,
                            inference_params.sequence_len_offset
                            * torch.ones(k.size(0), dtype=torch.int, device=k.device)
                            - empties,
                        ).unsqueeze(1)
                    else:
                        q = q.squeeze(1)
                        q = self.rotary_emb._single_forward(
                            q,
                            inference_params.sequence_len_offset
                            * torch.ones(q.size(0), dtype=torch.int, device=q.device)
                            - empties,
                        ).unsqueeze(1)
                        moved_k = k.clone()
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                moved_k[i][: -empties[i]] = k[i][empties[i] :]
                        moved_k = self.rotary_emb._single_eval_forward(moved_k, seqlen_offset=0)
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                k[i][empties[i] :] = moved_k[i][: -empties[i]]
                            else:
                                k[i] = moved_k[i]
                else:
                    q = self.rotary_emb._single_forward(q, inference_params.sequence_len_offset)
                    k = self.rotary_emb._single_forward(k, inference_params.sequence_len_offset)

                kv = torch.stack([k, v], dim=2)
                kv = _update_kv_cache(kv, inference_params, self.layer_idx)

            if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
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

                    if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            if total_q.dtype not in [torch.float16, torch.bfloat16]:
                                total_q = total_q.to(torch.bfloat16)
                            if total_kv.dtype not in [torch.float16, torch.bfloat16]:
                                total_kv = total_kv.to(torch.bfloat16)

                    output = flash_attn_unpadded_func(
                        total_q, total_kv, cu_seqlens, cu_seqlens, max_seqlen_q, max_seqlen_k, 0.0, None, True, False
                    ).to(x.dtype)

                    context = torch.zeros_like(q)
                    context = context.masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)

                else:
                    attn_mask = inference_params.attention_mask[:, -1, :].view(bsz, 1, 1, -1)

                    k, v = torch.chunk(kv, 2, dim=2)
                    k = k.squeeze(2)
                    v = v.squeeze(2)
                    sp = k.shape
                    scores = torch.einsum(
                        "blhd,bnhd->bhln",
                        q,
                        k.reshape(sp[0], sp[1], q.size(2), sp[3]),
                    ) / math.sqrt(q.size(-1))
                    scores = scores.masked_fill(attn_mask, -65000.0)
                    scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                    context = torch.einsum(
                        "bhmn,bnhd->bmhd",
                        scores,
                        v.reshape(sp[0], sp[1], q.size(2), sp[3]),
                    )
            else:
                context = self.inner_cross_attn(q, kv, causal=True)

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
