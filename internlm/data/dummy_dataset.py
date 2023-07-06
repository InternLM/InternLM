#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    """
    RandomDataset for generating random dataset.

    Args:
        num_samples (int): The number of samples to generate.
        max_len (int): The maximum length of each sample.

    """

    def __init__(self, num_samples=10000, max_len=1024) -> None:
        super().__init__()
        rng = np.random.RandomState(1999)
        max_num = rng.randint(1, 30, size=(num_samples,))
        rep_num = rng.randint(10, 200, size=(num_samples,))
        data = []
        lengths = []
        for n, r in zip(max_num, rep_num):
            d = list(range(n)) * r
            d = [n, r] + d
            d = d[:max_len]
            data.append(d)
            lengths.append(len(d))
        self.data = data
        self.max_len = max_len
        self.lengths = np.array(lengths, dtype=int)

    def __getitem__(self, index):
        d = self.data[index]
        input_ids = np.array(d, dtype=int)
        return {"tokens": list(input_ids), "type_id": 0}

    def get_dataset_name(self):
        return "dummy_path/dummy_lang/dummy_ds/train.bin"

    def __len__(self):
        return len(self.data)
