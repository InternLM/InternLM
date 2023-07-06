#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch


def packed_collate_fn(batch, packed_length):

    """
    Collate function for packed input sequences.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens", "labels", "type_ids", "cu_seqlens", and "indexes" keys.
        packed_length (int): The length of packed sequence.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
            "cu_seqlens", "indexes", and "type_ids" keys, and the tensor of padded "labels".

    Raises:
        AssertionError: If the length of a sample is not equal to packed_length.
        AssertionError: If the shape of the padded "input_ids" tensor does not have the correct shape.
    """

    xs, ys, cu_seqlens, indexes, ts = [], [], [], [], []
    for b in batch:
        assert (
            len(b["tokens"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['tokens'])} and {packed_length})"
        assert (
            len(b["labels"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['labels'])} and {packed_length})"
        assert (
            len(b["type_ids"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['type_ids'])} and {packed_length})"

        tokens = [abs(w) for w in b["tokens"]]
        labels = [w if w > 0 else -100 for w in b["labels"]]

        xs.append(torch.LongTensor(tokens))
        # The labels have been shifted here, so they are aligned with the output corresponding to the token
        ys.append(torch.LongTensor(labels))
        ts.append(torch.LongTensor(b["type_ids"]))
        cu_seqlens.append(torch.IntTensor(b["cu_seqlens"]))
        indexes.append(torch.LongTensor(b["indexes"]))

    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    indexes = torch.stack(indexes, dim=0)
    if len(set(map(len, cu_seqlens))) == 1:  # if has uniform length, then stack to save device transfer time
        cu_seqlens = torch.stack(cu_seqlens, dim=0)

    assert xs.shape[1] == packed_length, (xs.shape[1], packed_length)

    return {"input_ids": xs, "cu_seqlens": cu_seqlens, "indexes": indexes, "type_ids": ts}, ys


def jsonl_ds_collate_fn(batch, max_length_per_sample):
    """
    Collate function for json dataset.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens".
        max_length_per_sample (int): The length of output sequence.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
        and the tensor of padded "labels".

    """
    xs, ys = [], []
    for x in batch:
        x["tokens"] = x["tokens"][:max_length_per_sample]
        tokens = [abs(w) for w in x["tokens"]]
        labels = [w if w > 0 else -100 for w in x["tokens"]]
        labels = labels[1:] + [-100]
        xs.append(torch.as_tensor(tokens))
        ys.append(torch.as_tensor(labels))  # y has been shifted
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)

    xs = torch.cat([xs, xs.new_zeros(len(xs), max_length_per_sample - len(xs[0]))], dim=-1)
    ys = torch.cat([ys, ys.new_full((len(ys), max_length_per_sample - len(ys[0])), fill_value=-100)], dim=-1)

    return {"input_ids": xs}, ys
