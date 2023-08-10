import os
import sys 
import math
import re
from functools import partial
from typing import Optional

# from collections import namedtuple, OrderedDict
# from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp, FusedMLP, ParallelFusedMLP
from flash_attn.modules.embedding import ParallelGPT2Embeddings
try:
    from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.model.embedding import Embedding1D
from internlm.model.linear import (
    FeedForward,
    ScaleColumnParallelLinear,
)
from internlm.initialize.initialize_tensor import normal_, scaled_init_method_normal
from internlm.model.ssm.h3 import H3
from internlm.model.multi_head_attention import MHA
from internlm.model.utils import gather_forward_split_backward
from internlm.solver.pipeline_utils import partition_uniform
from internlm.utils.checkpoint import activation_checkpoint
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER

MODEL_TYPE = "H3"
logger = get_logger(__file__)

class SSMModelBlock(nn.Module):
    """
    1D SSMModel Flash Base Layer.

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
        hidden_size,
        mixer_cls: None,
        mlp_cls: None,
        norm_cls=nn.LayerNorm,
        drop_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        layer_idx: int = 0,
        residual_in_fp32: bool = False,
        dropout_selective_checkpoint: bool = True,
        checkpoint: bool = False,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx
        self.use_flash_attn = use_flash_attn
        
        self.mixer = mixer_cls(hidden_size)

        self.dropout1 = nn.Dropout(drop_rate)
    
        self.norm1 = norm_cls(hidden_size)
        self.norm2 = norm_cls(hidden_size)

        self.mlp = mlp_cls(hidden_size)
        
        self.dropout2 = nn.Dropout(drop_rate)
        self.use_swiglu = use_swiglu
        self.checkpoint = checkpoint
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
            # logger.error(f"this is: {os.getcwd()},{os.path.basename(__file__)},{sys._getframe().f_lineno}, PP_rank:{gpc.get_local_rank(ParallelMode.PIPELINE)} _residual.dtype:{_residual.dtype} _residual.float().dtype:{_residual.float().dtype} self.norm1.weight.dtype: {self.norm1.weight.dtype}")
            _hidden_states = self.norm1(_residual)
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_attn(hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if isinstance(self.mixer, MHA):
            if len(hidden_states.shape) == 3:
                hidden_states = hidden_states.squeeze(0)
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
        else:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states.unsqueeze(0)
            hidden_states = self.mixer(hidden_states, inference_params)

        def _dropout_and_norm_ffn(_residual, _hidden_states):
            _dropped = self.dropout2(_hidden_states)
            _residual = (_dropped + _residual) if _residual is not None else _dropped
            _hidden_states = self.norm2(_residual)
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_ffn, False, residual, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mlp(hidden_states)

        return hidden_states + residual


def create_mixer_cls(ssm_cls=H3, ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, layer_idx=None):
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        if attn_cfg:
            attn_cfg['layer_idx'] = layer_idx
        causal = True if attn_cfg is None else attn_cfg.pop('causal', True)
        layer_idx = True if attn_cfg is None else attn_cfg.pop('layer_idx', True)
        mixer_cls = partial(MHA, layer_idx=layer_idx, causal=causal, 
                            **(attn_cfg if attn_cfg is not None else {}))
    else:
        mixer_cls = partial(ssm_cls, layer_idx=layer_idx,
                            **(ssm_cfg if ssm_cfg is not None else {}))
    return mixer_cls

def create_block(hidden_size, 
                 ssm_cls=H3, 
                 norm_cls=nn.LayerNorm,
                 ssm_cfg=None, 
                 mlp_cfg=None,
                 attn_layer_idx=None,
                 attn_cfg=None,
                 drop_rate=0.0, 
                 residual_in_fp32=False,
                 layer_idx=None,
                 **kwargs):
    mixer_cls = create_mixer_cls(ssm_cls=ssm_cls, 
                                 ssm_cfg=ssm_cfg, 
                                 attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, 
                                 layer_idx=layer_idx)
    # mlp_cls = create_mlp_cls(hidden_size, d_inner=d_inner, fused_mlp=fused_mlp, mlp_cls=mlp_cls)
    mlp_cls = partial(ParallelFusedMLP, **(mlp_cfg if mlp_cfg is not None else {}))
    
    block = SSMModelBlock(hidden_size, 
                          mixer_cls, 
                          mlp_cls, 
                          norm_cls=norm_cls, 
                          drop_rate=drop_rate,
                          residual_in_fp32=residual_in_fp32,
                          **kwargs
                          )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)


class SSMModel(nn.Module):

    def __init__(
        self, 
        hidden_size: int, 
        num_layers: int, 
        vocab_size: int, 
        ssm_cfg=None,
        attn_layer_idx=None, 
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0, 
        max_position_embeddings=0,
        checkpoint: bool = False,
        mlp_ratio: int=4,
        # embed_dropout: float = 0.1, 
        # dropout_cls=nn.Dropout,
        norm_type=None,
        embed_grad_scale: float = 1.0, # Default to 1.0
        layer_norm_epsilon: float = 1e-5, 
        num_attention_heads: int=32,
        initializer_cfg=None,
        residual_in_fp32=False,
        first: bool = True,
        last: bool = True,
        start_layer_idx: int = 0,
        parallel_output: bool = True,
        embed_split_hidden: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float,
        # pass to SSMModelBlock
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.first = first
        self.last = last
        self.checkpoint = checkpoint
        norm_cls = partial(RMSNorm, eps=layer_norm_epsilon) if norm_type == 'rmsnorm' else partial(nn.LayerNorm, eps=layer_norm_epsilon)
        
        if self.checkpoint:
            raise NotImplementedError("Checkpointing not supported for h3 currently.")
        if first:
            if embed_split_hidden:
                self.embeddings = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
            else:
                self.embeddings = ParallelGPT2Embeddings(
                    embed_dim=hidden_size,
                    vocab_size=vocab_size,
                    max_position_embeddings=-1,
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    padding_idx=None,
                    sequence_parallel=gpc.config.model.sequence_parallel,
                    device=device,
                    dtype=dtype,
                )
            for _, param in self.embeddings.named_parameters():
                normal_(std=0.0052)(param)
                if gpc.get_world_size(ParallelMode.TENSOR) > 1:
                    setattr(param, IS_TENSOR_PARALLEL, True)
        self.embed_grad_scale = embed_grad_scale
        self.residual_in_fp32 = residual_in_fp32
        
        self.parallel_output = parallel_output
        
        head_dim = hidden_size // num_attention_heads
        attn_cfg = dict(
            num_heads=num_attention_heads,
            process_group=gpc.get_group(ParallelMode.TENSOR),
            dropout=attn_drop_rate,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            device=device,
            dtype=dtype,
        )
        mlp_cfg = dict(
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            activation="gelu_approx",
            process_group=gpc.get_group(ParallelMode.TENSOR),
            bias1=False,
            bias2=False,
            sequence_parallel=gpc.config.model.sequence_parallel,
            checkpoint_lvl=0,
            heuristic="auto",
            device=device,
            dtype=dtype,
        )#TODO
        self.layers = nn.ModuleList([create_block(
            hidden_size, 
            # d_inner=mlp_ratio * hidden_size, 
            ssm_cfg=ssm_cfg, 
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg, 
            layer_norm_epsilon=layer_norm_epsilon,
            residual_in_fp32=residual_in_fp32,
            layer_idx=i + start_layer_idx,
            mlp_cfg=mlp_cfg,
            norm_cls=norm_cls,
            drop_rate=drop_rate,
            # pass to SSMModelBlock
            checkpoint=checkpoint,
            dropout_selective_checkpoint=dropout_selective_checkpoint,
            use_scaled_init=use_scaled_init,
            use_swiglu=use_swiglu,
            use_flash_attn=use_flash_attn,
        ) for i in range(num_layers)])
        self.apply(partial(_init_weights, n_layer=num_layers,
                           **(initializer_cfg if initializer_cfg is not None else {})))
        
        if last:
            self.norm = norm_cls(hidden_size)
            self.head = ScaleColumnParallelLinear(
                in_features=hidden_size,
                out_features=vocab_size,
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
        
    def forward(self, hidden_states=None, input_ids=None, inference_params=None, cu_seqlens=None, indexes=None, max_seqlen=None):
        if self.first:
            hidden_states = self.embeddings(input_ids)
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
        
        for layer in self.layers:
            # logger.error(f"this is: {os.getcwd()} {os.path.basename(__file__)} {sys._getframe().f_lineno}, {gpc.get_local_rank(ParallelMode.PIPELINE)} layeridx{layer.layer_idx} {hidden_states.shape}")
            hidden_states = layer(hidden_states, 
                                    cu_seqlens=cu_seqlens,
                                    indexes=indexes,
                                    inference_params=inference_params,
                                    max_seqlen=max_seqlen,)
        
        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)
        if hasattr(self, "head"):
            hidden_states = self.head(hidden_states)            
        if not self.parallel_output:
            hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=-1)
        # cannot return two values
        return hidden_states


# class SSMLMHeadModel(nn.Module, GenerationMixin):

#     def __init__(self, hidden_size: int, n_layer: int, d_inner: int, vocab_size: int, ssm_cfg=None,
#                  attn_layer_idx=None, attn_cfg=None, max_position_embeddings=2048,
#                  resid_dropout: float = 0.0, embed_dropout: float = 0.1, dropout_cls=nn.Dropout,
#                  layer_norm_epsilon: float = 1e-5, initializer_cfg=None,
#                  fused_mlp=False, fused_dropout_add_ln=False, residual_in_fp32=False,
#                  pad_vocab_size_multiple: int = 1, **kwargs) -> None:
#         super().__init__()
#         if vocab_size % pad_vocab_size_multiple != 0:
#             vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
#         self.backbone = SSMModel(
#             hidden_size=hidden_size, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
#             ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
#             max_position_embeddings=max_position_embeddings,
#             resid_dropout=resid_dropout, embed_dropout=embed_dropout,
#             dropout_cls=dropout_cls, layer_norm_epsilon=layer_norm_epsilon,
#             initializer_cfg=initializer_cfg, fused_mlp=fused_mlp,
#             fused_dropout_add_ln=fused_dropout_add_ln, residual_in_fp32=residual_in_fp32, **kwargs
#         )
#         self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
#         # Initialize weights and apply final processing
#         self.apply(partial(_init_weights, n_layer=n_layer,
#                            **(initializer_cfg if initializer_cfg is not None else {})))
#         self.tie_weights()

#     def tie_weights(self):
#         self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

#     def forward(self, input_ids, position_ids=None, inference_params=None, last_token_only=False, attention_mask=None, labels=None):
#         hidden_states = self.backbone(input_ids, position_ids=position_ids,
#                                       inference_params=inference_params)
#         if last_token_only:
#             lm_logits = self.lm_head(hidden_states)[:, -1, :]
#         else:
#             lm_logits = self.lm_head(hidden_states)
#         CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
#         return CausalLMOutput(logits=lm_logits)

#     def load_state_dict(self, state_dict, strict=True):
#         # Remapping from our checkpoints that used different names
#         def key_mapping_backbone(key):
#             key = re.sub(r'^s4seq.encoder.', 'backbone.', key)
#             key = re.sub(r'^embedding.', 'backbone.embeddings.word_embeddings.', key)
#             key = re.sub(r'^backbone.norm', 'backbone.ln_0', key)
#             return key
#         state_dict = OrderedDict((key_mapping_backbone(k), v) for k, v in state_dict.items())
#         # Remapping from our checkpoints that used a different ordering of layers in the block
#         # Previous: Mixer / MLP -> Dropout -> Add -> LN
#         # Current: Dropout -> Add -> LN -> Attn / MLP
#         if 'backbone.ln_0.weight' in state_dict:
#             n_layers = len(self.backbone.layers)
#             ln_weight = state_dict.pop(f'backbone.layers.{n_layers - 1}.norm2.weight')
#             ln_bias = state_dict.pop(f'backbone.layers.{n_layers - 1}.norm2.bias')
#             state_dict['backbone.ln_f.weight'] = ln_weight
#             state_dict['backbone.ln_f.bias'] = ln_bias
#             for l in reversed(range(n_layers)):
#                 ln_weight = state_dict.pop(f'backbone.layers.{l}.norm1.weight')
#                 ln_bias = state_dict.pop(f'backbone.layers.{l}.norm1.bias')
#                 state_dict[f'backbone.layers.{l}.norm2.weight'] = ln_weight
#                 state_dict[f'backbone.layers.{l}.norm2.bias'] = ln_bias
#                 if l > 0:
#                     ln_weight = state_dict.pop(f'backbone.layers.{l - 1}.norm2.weight')
#                     ln_bias = state_dict.pop(f'backbone.layers.{l - 1}.norm2.bias')
#                     state_dict[f'backbone.layers.{l}.norm1.weight'] = ln_weight
#                     state_dict[f'backbone.layers.{l}.norm1.bias'] = ln_bias
#             ln_weight = state_dict.pop('backbone.ln_0.weight')
#             ln_bias = state_dict.pop('backbone.ln_0.bias')
#             state_dict[f'backbone.layers.0.norm1.weight'] = ln_weight
#             state_dict[f'backbone.layers.0.norm1.bias'] = ln_bias
#         return super().load_state_dict(state_dict, strict=strict)

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

    if kwargs["checkpoint"] is True:
        kwargs["checkpoint_fraction"] = 1.0
    else:
        kwargs["checkpoint_fraction"] = 0

    for start, end in parts:
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        # If there is no content in the final layer, assign the last layer.
        kwargs["last"] = end == num_layers and len(all_parts[-1]) != 0
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start
        chunk = SSMModel(**filter_kwargs(SSMModel.__init__, kwargs)).to(device)

        models.append(chunk)
    torch.distributed.barrier()
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model


@MODEL_INITIALIZER.register_module(module_name=MODEL_TYPE)
def build_h3_with_cfg(
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
    mlp_ratio=4.0,
    residual_in_fp32=False,
    norm_type="rmsnorm",
    drop_rate=0,
    attn_drop_rate=0,
    attn_layer_idx=None,
    ssm_cfg=None,
    apply_post_layer_norm=False,  # pylint: disable=W0613
    layer_norm_epsilon=1e-5,
    is_reward=False,
    dropout_selective_checkpoint=True,
    use_scaled_init: bool = True,
    use_swiglu: bool = True,
    use_flash_attn: bool = True,
    sequence_parallel: bool = False,  # pylint: disable=W0613
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
        attn_layer_idx (List[int]): Layer indexes of attention layers
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
        attn_layer_idx=attn_layer_idx,
        ssm_cfg=ssm_cfg,
    )

    return _build_generic_model_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)
