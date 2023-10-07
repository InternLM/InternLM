#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import warnings
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
from internlm.model.embedding import DynamicNTKScalingRotaryEmbedding, RotaryEmbedding
from internlm.model.linear import ColumnParallelLinearTorch, RowParallelLinearTorch, FSDPLinear

import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

import torch.distributed as dist


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = dist.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        # TODO Use all_to_all_single instead
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 0,
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    # def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any) -> Tensor:
    #     """ forward

    #     Arguments:
    #         query (Tensor): query input to the layer
    #         key (Tensor): key input to the layer
    #         value (Tensor): value input to the layer
    #         args: other args

    #     Returns:
    #         * output (Tensor): context output
    #     """
    #     # TODO Merge three alltoall calls into one
    #     #in shape : e.g.,  [s/p:h:]
    #     query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
    #     key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
    #     value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

    #     #out shape : e.g., [s:h/p:]
    #     context_layer = self.local_attn(query_layer, key_layer, value_layer, *args)

    #     output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

    #     #out e.g., [s/p::h]
    #     return output
    
    def forward(self, qkv: Tensor, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        #in shape : e.g.,  [s/p:h:]
        qkv = _SeqAllToAll.apply(self.spg, qkv, self.scatter_idx, self.gather_idx)
        # key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        # value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        #out shape : e.g., [s:h/p:]
        context_layer = self.local_attn(qkv, **kwargs)

        output = _SeqAllToAll.apply(self.spg, context_layer, 0, 2)

        #out e.g., [s/p::h]
        return output


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
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
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
                    scale_base=rotary_emb_scale_base,
                    device=device,
                    max_position_embeddings=max_position_embeddings,
                    scaling_factor=1.0,  # Currently do not support dynamic scaling.
                )
            else:
                self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device)

        # notice here should change bias=True
        # self.Wqkv = ColumnParallelLinearTorch(
        #     embed_dim,
        #     3 * embed_dim,
        #     process_group,
        #     bias=True,
        #     sequence_parallel=gpc.config.parallel.sequence_parallel,
        #     **factory_kwargs,
        # )  # according to https://spaces.ac.cn/archives/9577
        
        self.Wqkv = FSDPLinear(
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
        
        self.inner_attn_sp = DistributedAttention(self.inner_attn, sequence_process_group=process_group, scatter_idx=3, gather_idx=0)
        self.inner_cross_attn_sp = DistributedAttention(self.inner_cross_attn, sequence_process_group=process_group, scatter_idx=3, gather_idx=0)

        # output projection always have the bias (for now)
        # self.out_proj = RowParallelLinearTorch(
        #     embed_dim,
        #     embed_dim,
        #     process_group,
        #     sequence_parallel=gpc.config.parallel.sequence_parallel,
        #     **factory_kwargs,
        # )
        self.out_proj = FSDPLinear(
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

        if inference_params is None:
            if self.rotary_emb_dim > 0:
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
                if self.rotary_emb_dim > 0:
                    kwargs["inference_params"] = inference_params
                    qkv = self.rotary_emb(qkv, **kwargs)
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
                    # context = self.inner_attn(qkv, **kwargs).to(x.dtype)
                    context = self.inner_attn_sp(qkv, **kwargs).to(x.dtype)
            else:
                # context = self.inner_attn(qkv, **kwargs)
                context = self.inner_attn_sp(qkv, **kwargs)

        else:
            raise RuntimeError("Not support this right now")

        context = rearrange(context, "b h d -> b (h d)")  # recover the shape
        out = self.out_proj(context)
        return out
