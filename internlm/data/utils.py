#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

from internlm.core.context import global_context as gpc

DATASET_TYPE_IDS_MAP = {"en": 0, "cn": 1, "code": 2}


def get_dataset_type_id(path):
    import re

    match_idxes = []
    for key, idx in DATASET_TYPE_IDS_MAP.items():
        if re.search(rf"/[z_]*{key}/", path):
            match_idxes.append(idx)
    assert len(match_idxes) == 1, f"{path}, match_idxes should be 1, but got {match_idxes} from {DATASET_TYPE_IDS_MAP}"
    return match_idxes[0]


def unpack_data(input_ids, cu_seqlens):
    """
    input_ids: (n, packed_length)
    Return:
    output: (batch_size, max_length)
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

    if bsz == 1:
        outputs = outputs.squeeze(0)

    return outputs
