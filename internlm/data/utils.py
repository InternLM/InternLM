#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.nn.utils.rnn import pad_sequence

DATASET_TYPE_IDS_MAP = {"en": 0, "cn": 1, "code": 2, "ja": 3, "ar": 4, "kaoshi": 5}


def get_dataset_type_id(path):
    import re

    match_idxes = []
    for key, idx in DATASET_TYPE_IDS_MAP.items():
        if re.search(rf"/[z_]*{key}/", path):
            match_idxes.append(idx)
    assert len(match_idxes) == 1, f"{path}, match_idxes should be 1, but got {match_idxes} from {DATASET_TYPE_IDS_MAP}"
    return match_idxes[0]

def unpack_data(input_ids, cu_seqlens):
    '''
    input_ids: (1, packed_length)
    
    Return:
    output: (batch_size, max_length)
    '''
    if isinstance(cu_seqlens, list):
        assert len(cu_seqlens) == 1
        cu_seqlens = cu_seqlens[0]
    
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.squeeze(0)

    if isinstance(cu_seqlens, torch.Tensor):
        num_sequence = cu_seqlens.shape[0] - 1
    else:
        raise RuntimeError("The cu_seqlens should be list or torch.Tensor type")
    assert not num_sequence == 0
    # obtain the unpacked tensors
    
    # output = torch.zeros(num_sequence, max_lenth, device=input_ids.device, dtype=input_ids.dtype)
    tensor_list = []
    for i in range(num_sequence):
        tmp_tensor = input_ids[0, cu_seqlens[i]:cu_seqlens[i + 1]]
        tensor_list.append(tmp_tensor)
        # seq_length = cu_seqlens[i + 1] - cu_seqlens[i]
        # output[i, 0:seq_length] = input_ids[0, cu_seqlens[i]:cu_seqlens[i + 1]]
    
    output = pad_sequence(tensor_list, batch_first=True)
    return output
