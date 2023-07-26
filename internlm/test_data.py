import torch
import numpy as np
import itertools
from copy import deepcopy

sample_lengths = [10, 15, 27, 43, 51]
lengths = [10, 5, 12, 16, 8]
data = []

for i in range(5):
    t = list(range(0, sample_lengths[i]))
    t[0] = i
    data.append(t)

packed_length = 20
max_length = 8
micro_bsz = 2

print(data)

print()

def cal_pos(index):
    if index == 0:
        pre_pos = 0
    else:
        _, pre_pos = cal_pos(index - 1)
    pos = (index + 1) * micro_bsz
    return pre_pos, pos
        

def get_item(index):
    pack = []
    cu_seqlens = [0]
    
    pre_pos, pos = cal_pos(index)
    
    print(pre_pos)
    print(pos)
    
    while pre_pos < pos:
        sample = data[pre_pos]
        length = min(len(sample), max_length)
        chunk = sample[:length]
        pack.extend(chunk)
        cu_seqlens.append(cu_seqlens[-1] + len(chunk))
        pre_pos = pre_pos + 1
    
    if cu_seqlens[-1] != packed_length:
        pack = pack + [0] * (packed_length - cu_seqlens[-1])
        cu_seqlens.append(packed_length)
    return pack, cu_seqlens

pack, cu_seqlens = get_item(1)
print(pack)
print(cu_seqlens)