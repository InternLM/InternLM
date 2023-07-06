#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
A .bin file corresponds to a Dataset instance here.
"""

import json
import mmap
import os
import threading
from pathlib import Path

import numpy as np
import torch


class JsonlDataset(torch.utils.data.Dataset):
    """

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "tokens": List[int],
    }
    ```

    Note that only the "tokens" key is used.
    """

    def __init__(self, path: str, dataset_type_id: int = 0, min_length=50):
        self.path = path
        self.threadlocal = threading.local()
        resolved_path = Path(path).resolve()
        self.resolved_path = resolved_path
        self.meta = Path(f"{resolved_path}.meta")
        self.type_id = dataset_type_id

        # only build the cache in on the primary worker to prevent overloading nfs
        assert os.path.exists(self.meta), f"The cache file:{self.meta} is not found for file:{self.path}"
        try:
            with open(self.meta, "rb") as f:
                meta = np.load(f)
        except Exception as e:
            print(f"Cannot load file {self.meta}...")
            raise e
        self.offsets = meta[:, 0]
        self.lengths = meta[:, -1]

        if min_length > 0:
            mask = self.lengths >= min_length
            self.old_lengths = self.lengths.copy()
            self.old_length = len(self.offsets)
            self.offsets = self.offsets[mask]
            self.lengths = self.lengths[mask]

    def __getitem__(self, idx):
        f = self._get_mmap()
        position = self.offsets[idx]
        f.seek(position)
        item = f.readline().decode("utf-8")
        try:
            item = json.loads(item)
            item["length"] = len(item["tokens"])  # add a length info
            item["type_id"] = self.type_id
        except Exception as err:
            raise json.decoder.JSONDecodeError(
                doc=self.path,
                pos=position,
                msg=(
                    f"Error while loading JSONL line in file {self.path} at byte "
                    f"{position}. Contents of line:\n{item}\n{err}"
                ),
            )
        return item

    def get_dataset_name(self):
        return str(self.resolved_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            with open(self.path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if self.path.endswith(".gz") or self.path.endswith(".bz") or self.path.endswith(".bz2"):
                    raise NotImplementedError(
                        "Compressed files are not supported because .seek() would require "
                        "rereading the entire file, making performance too slow."
                    )
        return self.threadlocal.handles[-1]

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number if the number of documents
        # is not perfectly divisible by the data_subshard_count
        return len(self.offsets)
