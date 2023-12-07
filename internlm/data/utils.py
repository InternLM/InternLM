#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import re

import torch

from internlm.core.context import global_context as gpc


def get_dataset_type_ids_map(path):
    dirlist = list(os.listdir(path))
    dirlist.sort()
    return {key: idx for idx, key in enumerate(dirlist)}


def get_dataset_type_id(dataset_type_ids_map, path):
    match_idxes = []

    for key, idx in dataset_type_ids_map.items():
        if re.search(rf"/[z_]*{key}/", path):
            match_idxes.append(idx)
    assert len(match_idxes) == 1, f"{path}, match_idxes should be 1, but got {match_idxes} from {dataset_type_ids_map}"
    return match_idxes[0]


def unpack_data(input_ids, cu_seqlens, is_type_ids: bool = False):
    """
    input_ids: if input_ids is not type_ids, the shape is (1, packed_length)
               else the shape is (micro_num, packed_length)
    is_type_ids: whether the input_ids is type_ids

    Return:
    output: if input_ids is not type ids, the shape is (micro_bsz, max_length)
            else the shape is (micro_num, micro_bsz, max_length)
    """
    bsz = input_ids.shape[0]

    num_sequence = gpc.config.data["micro_bsz"]

    outputs = torch.zeros(bsz, num_sequence, gpc.config.data.seq_len, device=input_ids.device, dtype=input_ids.dtype)

    for i in range(bsz):
        output = torch.zeros(num_sequence, gpc.config.data.seq_len, device=input_ids.device, dtype=input_ids.dtype)
        cu_seqlens_slice = cu_seqlens[i]
        for j in range(num_sequence):
            seq_length = cu_seqlens_slice[j + 1] - cu_seqlens_slice[j]
            output[j, 0:seq_length] = input_ids[0, cu_seqlens_slice[j] : cu_seqlens_slice[j + 1]]
        outputs[i] = output

    # if the input_ids is not type_ids, we need squeeze the first dimension if it is 1.
    if bsz == 1 and not is_type_ids:
        outputs = outputs.squeeze(0)

    return outputs
