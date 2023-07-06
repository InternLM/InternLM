#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["SequenceGenerator"]


class InferenceParams:
    """
    Intermediate cache objects for inference
    """

    def __init__(
        self,
        max_sequence_len,
        max_batch_size,
        sequence_len_offset=0,
        batch_size_offset=0,
        key_value_memory_dict: dict = None,
        lengths_per_sample=None,
        attention_mask=None,
    ) -> None:

        self.max_sequence_len: int = max_sequence_len
        self.max_batch_size: int = max_batch_size
        self.sequence_len_offset: int = sequence_len_offset
        self.batch_size_offset: int = batch_size_offset
        if key_value_memory_dict is None:
            key_value_memory_dict = {}
        self.key_value_memory_dict: dict = key_value_memory_dict
        self.fused_ft_kernel: bool = False
        self.lengths_per_sample = lengths_per_sample
        self.attention_mask = attention_mask

    def reorder_state(self, indices):
        if self.lengths_per_sample is not None:
            self.lengths_per_sample = self.lengths_per_sample.index_select(index=indices, dim=0)
        for key, value in list(self.key_value_memory_dict.items()):
            value = value.index_select(index=indices, dim=0)
            self.key_value_memory_dict[key] = value


def _get_model_device(model):
    """
    obtain the device of an nn.Module.model

    Args:
        model: nn.Module

    Return: torch.device. if None, the parameters of this model is None.
    """
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


class SequenceGenerator:
    """
    Sequence Generator.
    """

    def __init__(self, decoder, eos_token_id, pad_token_id, bos_token_id):
        self.decoder = decoder
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    @torch.no_grad()
    def generate(
        self,
        tokens: "torch.LongTensor" = None,
        num_return_sequences=1,
        max_length: int = 20,
        num_beams: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
    ):
        """
        Args:
            tokens: the beginning tokens whose shape is [bsz, length]. If shape is None, default ''bos_token'' will be
                added to conduct generation.
            num_return_sequences: number of returned sequences.
            max_length: the max length of generated sequence.
            num_beams: the size of beam search.
            do_sample: whether using sample.
            temperature: it's meaningful when do_sample is True.
            top_k: sampling from top_k.
            top_p: sampling from top_p tokens(nucleus sampling).

        Return:
            the token sequence whose shape is [bsz, num_return_sequences, max_length]. If eos_token_id is not None,
                the ending of each sequence must be eos_token_id.
        """
        assert num_return_sequences <= num_beams, f"The `{num_return_sequences}` must be less than `{num_beams}`..."
        if do_sample:
            return sample_generate(
                self.decoder,
                tokens=tokens,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.eos_token_id,  # the ending token id
                pad_token_id=self.pad_token_id,
                repetition_penalty=repetition_penalty,  # the penalty degree for repetition tokens
                length_penalty=length_penalty,  # the penalty for length. if it > 1, then encourages long sequence.
                # Otherwise, encourages short sequence.
                bos_token_id=self.bos_token_id,
            )
        else:
            return greedy_generate(
                self.decoder,
                tokens=tokens,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                bos_token_id=self.bos_token_id,
            )


@torch.no_grad()
def greedy_generate(
    decoder,
    tokens=None,
    max_length=20,
    num_beams=1,
    num_return_sequences=1,
    eos_token_id=None,
    pad_token_id=0,
    repetition_penalty=1,
    length_penalty=1.0,
    bos_token_id=1,
    feat_mask=None,
    ffn_mask=None,
    layer_mask=None,
):
    """
    Search sequence greedily.

    Args:
        decoder: the Decoder object.
        tokens: the shape is [batch size, length]. If decoder is None, generating begins with bos_token_id.
        max_length: the max length for generated sequence.
        num_beams: the size of beam to decode.
        eos_token_id: the ending token id. If None, the decode length is max_length.
        pad_token_id: the token id of pad.
        repetition_penalty: the penalty degree for repetition tokens
        length_penalty: the penalty for length.

    """
    if num_beams == 1:
        token_ids = _no_beam_search_generate(
            decoder,
            tokens=tokens,
            max_length=max_length,
            temperature=1,
            top_k=50,
            top_p=1,
            eos_token_id=eos_token_id,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            feat_mask=feat_mask,
            ffn_mask=ffn_mask,
            layer_mask=layer_mask,
        )
    else:
        token_ids = _beam_search_generate(
            decoder,
            tokens=tokens,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=1,
            top_k=50,
            top_p=1,
            eos_token_id=eos_token_id,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            feat_mask=feat_mask,
            ffn_mask=ffn_mask,
            layer_mask=layer_mask,
        )

    return token_ids


@torch.no_grad()
def sample_generate(
    decoder,
    tokens,
    max_length=20,
    num_beams=1,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    eos_token_id=None,
    pad_token_id=0,
    repetition_penalty=1.0,
    length_penalty=1.0,
    bos_token_id=1,
):
    """
    generate sequence in sampling way.

    Args:
        decoder: the Decoder object.
        tokens: the shape is [batch size, length]. If decoder is None, generating begins with bos_token_id.
        max_length: the max length for generated sequence.
        num_beams: the size of beam to decode.
        num_return_sequences: number of returned sequence.
        temperature: annealing magnitude during sampling.
        top_k: sampling from top_k. (Default: 50)
        top_p: sampling from top_p tokens(nucleus sampling). (Default: 1.0)
        eos_token_id: the ending token id. If None, the decode length is max_length.
        pad_token_id: the token id of pad.
        repetition_penalty: the penalty degree for repetition tokens
        length_penalty: the penalty for length.

    """
    if num_beams == 1:
        token_ids = _no_beam_search_generate(
            decoder,
            tokens=tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
        )
    else:
        token_ids = _beam_search_generate(
            decoder,
            tokens=tokens,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
        )
    return token_ids


@torch.no_grad()
def _no_beam_search_generate(
    decoder,
    tokens,
    inference_params=None,
    max_length=20,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    eos_token_id=None,
    do_sample=True,
    repetition_penalty=1.0,
    length_penalty=1.0,
    pad_token_id=0,
    bos_token_id=1,
    feat_mask=None,
    ffn_mask=None,
    layer_mask=None,
):
    # delete num_return_sequences=1 for lint check;
    batch_size = tokens.size(0)
    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    has_bos = torch.all(tokens[:, 0].eq(bos_token_id))
    if has_bos:
        bos_pos = torch.where(tokens.eq(bos_token_id), 1, 0)
        bos_sum = bos_pos.cumsum(dim=-1)
        bos_pos = torch.where(bos_sum.eq(bos_sum[:, -1:]), 0, 1)
        to_atten_x = bos_pos[:, :, None]
        to_atten_y = bos_pos[:, None, :]
        # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
    else:
        bos_pos = torch.where(tokens.eq(bos_token_id), 1, 0)
        to_atten_x = bos_pos[:, :, None]
        to_atten_y = bos_pos[:, None, :]
        # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
    attention_mask = torch.logical_or(to_atten_x, to_atten_y).eq(1)
    if inference_params is None:
        inference_params = InferenceParams(
            max_sequence_len=max_length,
            max_batch_size=tokens.size(0),
            sequence_len_offset=0,
            batch_size_offset=0,
            key_value_memory_dict=None,
            lengths_per_sample=None,
            attention_mask=attention_mask,
        )

    if layer_mask is None:
        if feat_mask is None and ffn_mask is None:
            scores = decoder(**{"input_ids": tokens, "inference_params": inference_params})
        else:
            scores = decoder(
                **{
                    "input_ids": tokens,
                    "inference_params": inference_params,
                    "feat_mask": feat_mask,
                    "ffn_mask": ffn_mask,
                }
            )
    else:
        scores = decoder(
            **{
                "input_ids": tokens,
                "inference_params": inference_params,
                "feat_mask": feat_mask,
                "ffn_mask": ffn_mask,
                "layer_mask": layer_mask,
            }
        )

    if isinstance(scores, (list, tuple)):
        scores = scores[0]
    scores = scores[:, -1].float()
    inference_params.sequence_len_offset += tokens.size(1)
    if _eos_token_id != -1:
        scores[:, _eos_token_id] = -1e12
    next_tokens = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1)
    # tokens = tokens[:, -1:]

    real_max_length = max_length
    max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)

    while cur_len < real_max_length:
        # batch_size x vocab_size
        if has_bos:
            bos_pos = torch.where(token_ids.eq(bos_token_id), 1, 0)
            bos_sum = bos_pos.cumsum(dim=-1)
            bos_pos = torch.where(bos_sum.eq(bos_sum[:, -1:]), 0, 1)
            to_atten_x = bos_pos[:, :, None]
            to_atten_y = bos_pos[:, None, :]
            # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
        else:
            bos_pos = torch.where(token_ids.eq(bos_token_id), 1, 0)
            to_atten_x = bos_pos[:, :, None]
            to_atten_y = bos_pos[:, None, :]
            # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
        attention_mask = torch.logical_or(to_atten_x, to_atten_y).eq(1)
        inference_params.attention_mask = attention_mask
        if layer_mask is None:
            if feat_mask is None and ffn_mask is None:
                scores = decoder(**{"input_ids": token_ids[:, -1:], "inference_params": inference_params})
            else:
                scores = decoder(
                    **{
                        "input_ids": token_ids[:, -1:],
                        "inference_params": inference_params,
                        "feat_mask": feat_mask,
                        "ffn_mask": ffn_mask,
                    }
                )
        else:
            scores = decoder(
                **{
                    "input_ids": token_ids[:, -1:],
                    "inference_params": inference_params,
                    "feat_mask": feat_mask,
                    "ffn_mask": ffn_mask,
                    "layer_mask": layer_mask,
                }
            )

        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        scores = scores[:, -1].float()
        inference_params.sequence_len_offset += 1

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = (
                lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            )
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id is not None and length_penalty != 1.0:
            # batch_size x vocab_size
            token_scores = scores / cur_len**length_penalty
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)

            scores = scores.masked_scatter(eos_mask, token_scores)

        if do_sample:
            if temperature > 0 and temperature != 1:
                scores = scores / temperature

            scores = top_k_top_p_filtering(scores, top_k, top_p, min_tokens_to_keep=2)
            # add 1e-12 to avoid https://github.com/pytorch/pytorch/pull/27523
            probs = F.softmax(scores, dim=-1) + 1e-12

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # batch_size
        else:
            next_tokens = torch.argmax(scores, dim=-1)  # batch_size

        if _eos_token_id != -1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len + 1), _eos_token_id)
        next_tokens = next_tokens.masked_fill(dones, pad_token_id)
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        if dones.min() == 1:
            break

    # if eos_token_id is not None:
    #     # setting the eos at the maximum length position
    #     tokens.scatter(index=max_lengths[:, None], dim=1, value=eos_token_id)
    # if cur_len == max_length:
    #     # If eos is not reached by the maximum length, forcibly replace the last word with eos
    #     token_ids[:, -1].masked_fill_(~dones, eos_token_id)
    # TODO Here we are simply adding an extra dimension for interface compatibility, but in the future it will need to
    # be able to return multiple real results
    return token_ids[:, None]


@torch.no_grad()
def _beam_search_generate(
    decoder,
    tokens,
    inference_params=None,
    max_length=20,
    num_beams=4,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    eos_token_id=None,
    do_sample=True,
    repetition_penalty=1.0,
    length_penalty=1.0,
    pad_token_id=0,
    bos_token_id=1,
    feat_mask=None,
    ffn_mask=None,
    layer_mask=None,
) -> torch.LongTensor:

    device = _get_model_device(decoder)
    batch_size = tokens.size(0)

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    has_bos = torch.all(tokens[:, 0].eq(bos_token_id))

    if has_bos:
        bos_pos = torch.where(tokens.eq(bos_token_id), 1, 0)
        bos_sum = bos_pos.cumsum(dim=-1)
        bos_pos = torch.where(bos_sum.eq(bos_sum[:, -1:]), 0, 1)
        to_atten_x = bos_pos[:, :, None]
        to_atten_y = bos_pos[:, None, :]
        # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
    else:
        bos_pos = torch.where(tokens.eq(bos_token_id), 1, 0)
        to_atten_x = bos_pos[:, :, None]
        to_atten_y = bos_pos[:, None, :]
        # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
    attention_mask = torch.logical_or(to_atten_x, to_atten_y).eq(1)

    if inference_params is None:
        inference_params = InferenceParams(
            max_sequence_len=max_length,
            max_batch_size=tokens.size(0),
            sequence_len_offset=0,
            batch_size_offset=0,
            key_value_memory_dict=None,
            lengths_per_sample=None,
            attention_mask=attention_mask,
        )

    if layer_mask is None:
        if feat_mask is None and ffn_mask is None:
            scores = decoder(**{"input_ids": tokens, "inference_params": inference_params})
        else:
            scores = decoder(
                **{
                    "input_ids": tokens,
                    "inference_params": inference_params,
                    "feat_mask": feat_mask,
                    "ffn_mask": ffn_mask,
                }
            )
    else:
        scores = decoder(
            **{
                "input_ids": tokens,
                "inference_params": inference_params,
                "feat_mask": feat_mask,
                "ffn_mask": ffn_mask,
                "layer_mask": layer_mask,
            }
        )

    if isinstance(scores, (list, tuple)):
        scores = scores[0]
    scores = scores[:, -1].float()
    inference_params.sequence_len_offset += tokens.size(1)
    if _eos_token_id != -1:
        scores[:, _eos_token_id] = -1e12
    vocab_size = scores.size(1)
    assert vocab_size >= num_beams, "num_beams should be smaller than " "the number of vocabulary size."

    if do_sample:
        probs = F.softmax(scores, dim=-1) + 1e-12
        # (batch_size, num_beams)
        next_tokens = torch.multinomial(probs, num_samples=num_beams)
        logits = probs.log()
        # (batch_size, num_beams)
        next_scores = logits.gather(dim=1, index=next_tokens)
    else:
        scores = F.log_softmax(scores, dim=-1)  # (batch_size, vocab_size)
        # obtain (batch_size, num_beams), (batch_size, num_beams)
        next_scores, next_tokens = torch.topk(scores, num_beams, dim=1, largest=True, sorted=True)

    indices = torch.arange(batch_size, dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_beams)
    inference_params.reorder_state(indices)

    # batch_size * num_beams x length
    tokens = tokens.index_select(dim=0, index=indices)
    # genrated token (batch_size', cur_len)
    token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
    dones = [False] * batch_size

    beam_scores = next_scores.view(-1)  # batch_size * num_beams

    cur_len = token_ids.size(1)

    real_max_length = max_length
    max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)
    hypos = [
        BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
    ]
    # 0, num_beams, 2*num_beams, ...
    batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1).to(token_ids)

    while cur_len < real_max_length:
        if has_bos:
            bos_pos = torch.where(token_ids.eq(bos_token_id), 1, 0)
            bos_sum = bos_pos.cumsum(dim=-1)
            bos_pos = torch.where(bos_sum.eq(bos_sum[:, -1:]), 0, 1)
            to_atten_x = bos_pos[:, :, None]
            to_atten_y = bos_pos[:, None, :]
            # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
        else:
            bos_pos = torch.where(token_ids.eq(bos_token_id), 1, 0)
            to_atten_x = bos_pos[:, :, None]
            to_atten_y = bos_pos[:, None, :]
            # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
        attention_mask = torch.logical_or(to_atten_x, to_atten_y).eq(1)

        inference_params.attention_mask = attention_mask
        # (bsz x num_beams, vocab_size)

        if layer_mask is None:
            if feat_mask is None and ffn_mask is None:
                scores = decoder(**{"input_ids": token_ids[:, -1:], "inference_params": inference_params})
            else:
                scores = decoder(
                    **{
                        "input_ids": token_ids[:, -1:],
                        "inference_params": inference_params,
                        "feat_mask": feat_mask,
                        "ffn_mask": ffn_mask,
                    }
                )
        else:
            scores = decoder(
                **{
                    "input_ids": token_ids[:, -1:],
                    "inference_params": inference_params,
                    "feat_mask": feat_mask,
                    "ffn_mask": ffn_mask,
                    "layer_mask": layer_mask,
                }
            )

        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        scores = scores[:, -1].float()
        inference_params.sequence_len_offset += 1
        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = (
                lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            )
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if _eos_token_id != -1:
            max_len_eos_mask = max_lengths.eq(cur_len + 1)
            eos_scores = scores[:, _eos_token_id]
            scores[:, _eos_token_id] = torch.where(max_len_eos_mask, eos_scores + 1e32, eos_scores)

        if do_sample:
            if temperature > 0 and temperature != 1:
                scores = scores / temperature

            scores = top_k_top_p_filtering(scores, top_k, top_p, min_tokens_to_keep=num_beams + 1)
            # add 1e-12 to avoid https://github.com/pytorch/pytorch/pull/27523
            probs = F.softmax(scores, dim=-1) + 1e-12

            # batch_size' x (num_beams+1)
            _tokens = torch.multinomial(probs, num_samples=num_beams + 1)

            logits = probs.log()
            # batch_size' x (num_beams+1)
            _scores = logits.gather(dim=1, index=_tokens)
            # batch_size' x (num_beams+1)
            _scores = _scores + beam_scores[:, None]
            _scores = _scores.view(batch_size, num_beams * (num_beams + 1))
            next_scores, ids = _scores.topk(2 * num_beams, dim=1, largest=True, sorted=True)
            _tokens = _tokens.view(batch_size, num_beams * (num_beams + 1))
            # (batch_size, 2*num_beams)
            next_tokens = _tokens.gather(dim=1, index=ids)
            # (batch_size, 2*num_beams)
            from_which_beam = torch.floor(ids.float() / (num_beams + 1)).long()
        else:
            # (batch_size * num_beams, vocab_size)
            scores = F.log_softmax(scores, dim=-1)
            # (batch_size * num_beams, vocab_size)
            _scores = scores + beam_scores[:, None]
            # (batch_size, num_beams*vocab_size)
            _scores = _scores.view(batch_size, -1)
            # (bsz, 2*num_beams)
            next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            # (batch_size, 2*num_beams)
            from_which_beam = torch.floor(ids.float() / vocab_size).long()
            next_tokens = ids % vocab_size  # (batch_size, 2*num_beams)

        # next_scores, sorted_inds = next_scores.sort(dim=-1, descending=True)
        # next_tokens = next_tokens.gather(dim=1, index=sorted_inds)
        # from_which_beam = from_which_beam.gather(dim=1, index=sorted_inds)

        not_eos_mask = next_tokens.ne(_eos_token_id)
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)
        keep_mask = not_eos_mask.__and__(keep_mask)

        _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
        _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams)
        _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
        beam_scores = _next_scores.view(-1)

        flag = True
        if cur_len + 1 == real_max_length:
            eos_batch_idx = torch.arange(batch_size).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
            eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(batch_size)
            eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)
        else:
            effective_eos_mask = next_tokens[:, :num_beams].eq(_eos_token_id)  # batch_size x num_beams
            if effective_eos_mask.sum().gt(0):
                eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]
            else:
                flag = False

        if flag:
            _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
            for batch_idx, beam_ind, beam_idx in zip(
                eos_batch_idx.tolist(), eos_beam_ind.tolist(), eos_beam_idx.tolist()
            ):
                if not dones[batch_idx]:
                    score = next_scores[batch_idx, beam_ind].item()
                    if _eos_token_id != -1:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                    else:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx].clone(), score)

        reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)
        inference_params.reorder_state(reorder_inds)
        token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)

        for batch_idx in range(batch_size):
            dones[batch_idx] = (
                dones[batch_idx]
                or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item())
                or max_lengths[batch_idx * num_beams] == cur_len + 1
            )

        cur_len += 1

        if all(dones):
            break

    # select the best hypotheses
    tgt_len = token_ids.new_zeros(batch_size, num_return_sequences)
    best = []

    for i, hypotheses in enumerate(hypos):
        # best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        sorted_hyp = list(sorted(hypotheses.hyp, key=lambda x: x[0], reverse=True))
        _best = []
        for j, hyp in zip(range(num_return_sequences), sorted_hyp):
            hyp = hyp[1]
            if _eos_token_id != -1:
                hyp = torch.cat([hyp, token_ids.new_ones(1) * _eos_token_id])
            tgt_len[i, j] = len(hyp)
            _best.append(hyp)
        best.append(_best)

    # generate target batch
    decoded = token_ids.new_zeros(batch_size, num_return_sequences, tgt_len.max().item()).fill_(pad_token_id)
    for i, hypo in enumerate(best):
        for j, _hypo in enumerate(hypo):
            decoded[i, j, : tgt_len[i, j]] = _hypo

    return decoded


class BeamHypotheses(object):
    """
    BeamHypotheses
    """

    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """Initialize n-best list of hypotheses."""
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """Number of hypotheses in the list."""
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """Add a new hypothesis to the list."""
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """If there are enough hypotheses and that none of the hypotheses being
        generated can become better than the worst one in the heap, then we are
        done with this sentence."""
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length**self.length_penalty


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    Based on the values of top_k and top_p, set the values that do not meet the criteria to the filter_value.

    Args:
        logits: logit value, shape is [bsz, vocab_size].
        top_k: If it is greater than 0, only the probabilities of the top_k vocabulary are kept, and the rest of
            the positions are set to filter_value.
        top_p: according to http://arxiv.org/abs/1904.09751.
        filter_value: filter value
        min_tokens_to_keep: The probability of words in each sample‘s returned distribution will not be
            lower than this value.

    """
    if top_k > 0:
        # Safety check
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Remove all tokens with a probability less than the last token of
        # the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            # (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
