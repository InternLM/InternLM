from typing import Optional, Union  # For comments

import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_kvpacked_func
from flash_attn.flash_attn_interface import FlashAttnVarlenKVPackedFunc
from flash_attn.modules.mha import _update_kv_cache
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
from torch import nn

from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.embedding import RotaryEmbedding


class OneDConvertedParallelMHA2(nn.Module):
    """
    Multi-head self-attention and cross-attention
    The shape of causal attention matrix is total * hiddensize.

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
        use_flash_attn (boolean): Whether to use flash attention or not. If False, vanilla attention module
                                    will be used. False by default.
        checkpointing (boolean): Whether to use torch.utils.checkpointing to save VRAM or not. False by default.
        sequence_parallel (boolean): If True, we're doing Tensor Parallel with sequence parallelism. An all_gather_raw
                                    of x will be done before doing the matmul.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.

    Raises:
        ImportError: An ImportError is raised if ColumnParallelLinear or RowParallelLinear is None
        RuntimeError: An RuntimeError is raised if the inference_params is not None when calling _packed_forward

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
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        use_flash_attn: bool = False,
        checkpointing: bool = False,
        sequence_parallel: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        rot_embed_HF_impl: Optional[bool] = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing
        self.rot_embed_HF_impl = rot_embed_HF_impl

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device)

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")

        self.kv_dim = self.head_dim * num_kv_heads
        self.num_kv_heads = num_kv_heads

        self.wq = ColumnParallelLinear(
            embed_dim, embed_dim, process_group, bias=bias, sequence_parallel=sequence_parallel, **factory_kwargs
        )

        self.wk = ColumnParallelLinear(
            embed_dim, self.kv_dim, process_group, bias=bias, sequence_parallel=sequence_parallel, **factory_kwargs
        )

        self.wv = ColumnParallelLinear(
            embed_dim, self.kv_dim, process_group, bias=bias, sequence_parallel=sequence_parallel, **factory_kwargs
        )

        # assert use_flash_attn
        # inner_cross_attn_cls = FlashCrossAttention

        # self.inner_cross_attn = inner_cross_attn_cls(
        #     causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        # )

        self.inner_cross_attn_causal = causal
        self.inner_cross_attn_softmax_scale = softmax_scale
        self.inner_cross_attn_dropout = dropout

        # output projection always have the bias (for now)
        self.wo = RowParallelLinear(
            embed_dim, embed_dim, process_group, sequence_parallel=sequence_parallel, bias=bias, **factory_kwargs
        )

        # need to assign tp attribute so that colossalai know it is tensor parallel module
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["wo", "wq", "wk", "wv"]:
                for param in getattr(self, name).parameters():
                    setattr(param, IS_TENSOR_PARALLEL, True)

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get("indexes", None) is not None:
            return self._packed_forward(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._forward(x=x, seqlen=seqlen, inference_params=inference_params)

    def _forward(self, x, seqlen=None, inference_params=None):
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

        # qkv shift
        # the rotary embedding in flash attention module in performed by separating the front and back parts, while
        # most of others are done by odd-even methods.
        if not self.rot_embed_HF_impl:
            q = torch.cat([q[..., ::2], q[..., 1::2]], dim=-1)
            k = torch.cat([k[..., ::2], k[..., 1::2]], dim=-1)

        if inference_params is None:
            if self.rotary_emb_dim > 0:
                q = self.rotary_emb.single_eval_forward(q)
                k = self.rotary_emb.single_eval_forward(k)
            if not self.checkpointing:
                kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
                context = self.inner_cross_attn(q, kv)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_cross_attn, q, kv)

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
                    moved_q = self.rotary_emb.single_eval_forward(
                        moved_q, seqlen_offset=inference_params.sequence_len_offset
                    )
                    moved_k = self.rotary_emb.single_eval_forward(
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
                    q = self.rotary_emb.single_forward(
                        q,
                        inference_params.sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device)
                        - empties,
                    ).unsqueeze(1)
                    k = self.rotary_emb.single_forward(
                        k,
                        inference_params.sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device)
                        - empties,
                    ).unsqueeze(1)
            else:
                q = self.rotary_emb.single_eval_forward(q, seqlen_offset=inference_params.sequence_len_offset)
                k = self.rotary_emb.single_eval_forward(k, seqlen_offset=inference_params.sequence_len_offset)

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

            causal = None if inference_params.sequence_len_offset == 0 else False
            if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
                if inference_params.sequence_len_offset == 0:  # 第一次进入，是个方阵
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

                    output = FlashAttnVarlenKVPackedFunc.apply(
                        total_q, total_kv, cu_seqlens, cu_seqlens, max_seqlen_q, max_seqlen_k, 0.0, None, True, False
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
                    import math

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
        q = self.rotary_emb.single_forward(q, indexes)
        k = self.rotary_emb.single_forward(k, indexes)

        if inference_params is None:
            if not self.checkpointing:
                kv = torch.concat([k.unsqueeze(1), v.unsqueeze(1)], dim=1)
                # context = self.inner_cross_attn(q, kv, **kwargs, cu_seqlens_k=kwargs['cu_seqlens'], max_seqlen_k=kwargs['max_seqlen'])
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
                context = torch.utils.checkpoint.checkpoint(
                    flash_attn_varlen_kvpacked_func,
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


class FeedForward(nn.Module):
    """
    FeedForward, use SwiGLU by default.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.

    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )
        self.w3 = ColumnParallelLinear(
            in_features, hidden_features, process_group, bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.w2 = RowParallelLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )
        # Add tensor parallel attribute so that colossalai know it is a tensor parallel module.
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["w1", "w2", "w3"]:
                for param in getattr(self, name).parameters():
                    setattr(param, IS_TENSOR_PARALLEL, True)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """
    RMS Normarlization.

    Args:
        dim (int): the dimention of model.
        eps (float): bias term. 1e-6 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.

    """

    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)
