#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import itertools as it
import operator
import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from internlm.core.context import global_context as gpc
from internlm.data.single_dataset import JsonlDataset
from internlm.data.utils import get_dataset_type_id
from internlm.utils.logger import get_logger

DEFAULT_SEED = 1024
logger = get_logger(__file__)


class PackedDataset(torch.utils.data.Dataset):
    """
    The class PackedDataset takes in a dataset and aggregates samples of different
    lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        max_length_per_sample: The maximum length of each original sample. Default is 2048.
        packed_length: The length of each packed sample. Default is 4096.
    """

    def __init__(
        self,
        dataset,
        max_length_per_sample: int = 2048,
        packed_length: int = 4096,
    ):
        assert hasattr(dataset, "lengths")
        assert len(getattr(dataset, "lengths")) == len(
            dataset
        ), "The dataset must have lengths attribute and have the same length as the dataset"
        self.dataset = dataset
        self.max_length_per_sample = max_length_per_sample
        self.lengths = getattr(self.dataset, "lengths")
        self.packed_length = packed_length
        # Force a seed to be fixed to prevent problems caused by the seed not being restored when restarting

        self.seed = DEFAULT_SEED
        self.sample_indices, self.len_samples_shuffled, self.acm_len_samples = self.accu_sample_len(seed=self.seed)
        self.num_tokens = sum(self.lengths)

    def get_dataset_name(self):
        return self.dataset.get_dataset_name()

    def accu_sample_len(self, seed=None):
        """accumulative length of samples"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(self.seed - 1)

        sample_indices = np.arange(len(self.lengths))
        rng.shuffle(sample_indices)
        len_samples_shuffled = list(map(self.lengths.__getitem__, sample_indices))
        acm_len_samples = list(it.accumulate(len_samples_shuffled, operator.add))
        return sample_indices, len_samples_shuffled, acm_len_samples

    def __len__(self):
        # Line 405 of document_to_sequence.py in metaseq is directly spliced,
        # without additional consideration of sos or eos
        n_packs = self.num_tokens // self.packed_length
        return n_packs

    def cal_map(self, carriage_idx: int = 0):
        assert carriage_idx >= 0
        length_train = (carriage_idx + 1) * self.packed_length
        post_pos = np.searchsorted(self.acm_len_samples, length_train, side="left")
        return post_pos

    def mapping(self, pack_idx: int = 0):
        # pack_idx is zero-based
        pre_pos, pre_token_id = 0, 0
        if pack_idx > 0:
            pre_pos = self.cal_map(pack_idx - 1)
            pre_token_id = self.len_samples_shuffled[pre_pos] - (
                self.acm_len_samples[pre_pos] - (pack_idx) * self.packed_length
            )
            if pre_token_id == self.len_samples_shuffled[pre_pos]:
                pre_pos += 1
                pre_token_id = 0

        pos = self.cal_map(pack_idx)
        token_id = self.len_samples_shuffled[pos] - (self.acm_len_samples[pos] - (pack_idx + 1) * self.packed_length)
        return pre_pos, pre_token_id, pos, token_id

    def build_pack(self, pre_pos: int, pre_token_id: int, pos: int, token_id: int):
        pack, cu_seqlens, indexes, labels, type_ids = [], [0], [], [], []

        while pre_pos < pos:
            sample_idx = self.sample_indices[pre_pos]
            sample = self.dataset[sample_idx]
            chunk = sample["tokens"][pre_token_id:]
            pack.extend(chunk)
            _labels = deepcopy(chunk)
            _labels = list(_labels[1:]) + [-100]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            type_ids.extend([sample.get("type_id", 0)] * len(chunk))
            num_new_samples, tokens_left = divmod(len(chunk), self.max_length_per_sample)
            for _ in range(num_new_samples):
                cu_seqlens.append(cu_seqlens[-1] + self.max_length_per_sample)
                indexes.extend(list(range(self.max_length_per_sample)))
            if tokens_left > 0:
                cu_seqlens.append(cu_seqlens[-1] + tokens_left)
                indexes.extend(list(range(tokens_left)))
            pre_pos = pre_pos + 1
            pre_token_id = 0

        sample_idx = self.sample_indices[pos]
        sample = self.dataset[sample_idx]
        chunk = sample["tokens"][pre_token_id:token_id]  # fragement of a sample
        pack.extend(chunk)
        _labels = deepcopy(chunk)
        if token_id == len(sample["tokens"]):
            _labels = list(_labels[1:]) + [-100]
        else:
            if token_id > len(sample["tokens"]):
                print(f"token_id {token_id}, len of sample {len(sample['tokens'])}")
            _labels = list(_labels[1:]) + [sample["tokens"][token_id]]
        assert len(_labels) == len(chunk), (_labels, chunk)
        labels.extend(_labels)
        type_ids.extend([sample.get("type_id", 0)] * len(chunk))
        num_new_samples, tokens_left = divmod(len(chunk), self.max_length_per_sample)
        for _ in range(num_new_samples):
            cu_seqlens.append(cu_seqlens[-1] + self.max_length_per_sample)
            indexes.extend(list(range(self.max_length_per_sample)))
        if tokens_left > 0:
            cu_seqlens.append(cu_seqlens[-1] + tokens_left)
            indexes.extend(list(range(tokens_left)))

        out = {"tokens": pack, "cu_seqlens": cu_seqlens, "indexes": indexes, "labels": labels, "type_ids": type_ids}
        return out

    def cal_pos_unpack(self, index):
        if index == 0:
            pre_pos = 0
        else:
            pre_pos = index * gpc.config.data["micro_bsz"]

        pos = (index + 1) * gpc.config.data["micro_bsz"]
        return pre_pos, pos

    def build_unpack(self, index):

        pre_pos, pos = self.cal_pos_unpack(index)

        pack, cu_seqlens, indexes, labels, type_ids = [], [0], [], [], []

        while pre_pos < pos and pre_pos < len(self.dataset):
            sample_idx = self.sample_indices[pre_pos]
            sample = self.dataset[sample_idx]
            length = min(len(sample["tokens"]), self.max_length_per_sample)
            chunk = sample["tokens"][0:length]
            pack.extend(chunk)
            _labels = deepcopy(chunk)
            _labels = list(_labels[1:]) + [-100]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            type_ids.extend([sample.get("type_id", 0)] * len(chunk))
            cu_seqlens.append(cu_seqlens[-1] + len(chunk))
            indexes.extend(list(range(length)))
            pre_pos = pre_pos + 1

        if cu_seqlens[-1] != self.packed_length:
            pack = pack + [0] * (self.packed_length - cu_seqlens[-1])
            labels = labels + [0] * (self.packed_length - cu_seqlens[-1])
            type_ids = type_ids + [0] * (self.packed_length - cu_seqlens[-1])
            indexes.extend(list(range(self.packed_length - cu_seqlens[-1])))
            cu_seqlens.append(self.packed_length)

        assert len(pack) == self.packed_length

        out = {"tokens": pack, "cu_seqlens": cu_seqlens, "indexes": indexes, "labels": labels, "type_ids": type_ids}
        return out

    def __getitem__(self, item: int) -> Dict:
        """Given the index, it returns a dict as
        {
         'tokens': List[int],
         'cu_seqlens': List[int],
         'indexes': List[int], # denotes positional vector as 'tokens'
         'labels': List[int], # corresponds to 'tokens' and shifted yet, -100 means skipping prediction
        }
        """

        if gpc.config.model.use_flash_attn:
            pos_before, token_id_before, pos_after, token_id_after = self.mapping(item)
            return self.build_pack(pos_before, token_id_before, pos_after, token_id_after)

        return self.build_unpack(item)


class PackedDatasetWithoutCuSeqlen(torch.utils.data.Dataset):
    """
    A dataset wrapper that aggregates samples with different lengths based on packed_length.
    If a sample is shorter than max_length_per_sample, it will be merged with other samples.
    For example, given a dataset with 10 samples:
    [1, 2, 3, 4, 5]
    [6, 7]
    [8, 9, 10, 11]
    [12, ..., 100]
    ...

    Args:
        dataset: The original dataset to be wrapped.
        max_length_per_sample (int): The maximum length allowed for each sample.
        packed_length (int): The desired length for each packed sample.
    """

    def __init__(
        self,
        dataset,
        max_length_per_sample: int = 2048,
        packed_length: int = 4096,
        debug=False,
    ):
        assert packed_length % max_length_per_sample == 0
        assert hasattr(dataset, "lengths")
        assert len(getattr(dataset, "lengths")) == len(
            dataset
        ), "The dataset must have lengths attribute and have the same length as the dataset"
        self.dataset = dataset
        self.max_length_per_sample = max_length_per_sample
        self.lengths = getattr(self.dataset, "lengths")
        self.bsz = packed_length // max_length_per_sample
        self.packed_length = packed_length
        self.debug = debug
        # Force a seed to be fixed to prevent problems caused by the seed not being restored when restarting

        self.seed = DEFAULT_SEED
        indices = np.arange(len(self.lengths))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)
        self.indices = indices
        self.cum_lens = np.cumsum(self.lengths[self.indices])
        self.num_tokens = sum(self.lengths)

    def get_dataset_name(self):
        return self.dataset.get_dataset_name()

    def __len__(self):
        n_packs = self.num_tokens // self.packed_length
        return n_packs

    def find_offset(self, offset):
        idx = np.searchsorted(self.cum_lens, offset, side="right")
        if idx == 0:
            return idx, offset
        length = offset - self.cum_lens[idx - 1]
        return idx, length

    def pdebug(self, line):
        if self.debug:
            print(line, flush=True)

    def __getitem__(self, item: int) -> Dict:
        """Given the index, it returns a dict as
        {
         'tokens': List[int],
         'cu_seqlens': List[int],
         'indexes': List[int], # denotes positional vector as 'tokens'
         'labels': List[int], # corresponds to 'tokens' and shifted yet, -100 means skipping prediction
        }
        """

        start_idx, start_length = self.find_offset(item * self.packed_length)
        end_idx, end_length = self.find_offset((item + 1) * self.packed_length)
        pack_tokens = []
        pack_labels = []
        type_ids = []

        self.pdebug(f"item : {item}, start_idx:{start_idx}, start_length:{start_length} ")
        self.pdebug(f"item : {item}, end_idx:{end_idx}, end_length:{end_length} ")

        if start_idx == end_idx:
            idx = self.indices[start_idx]
            sample = self.dataset[idx]
            self.pdebug(f"item : {item}, idx: {idx}, len : {len(sample['tokens'])}")
            tokens = sample["tokens"][start_length:end_length]
            pack_tokens.extend(tokens)
            pack_labels.extend(tokens[1:] + [-100])
            type_ids.extend([sample["type_id"]] * len(tokens))
            return {
                "tokens": pack_tokens,
                "cu_seqlens": [i * self.max_length_per_sample for i in range(self.bsz + 1)],
                "indexes": list(range(self.max_length_per_sample)) * self.bsz,
                "labels": pack_labels,
                "type_ids": type_ids,
            }

        idx = self.indices[start_idx]
        sample = self.dataset[idx]
        self.pdebug(f"item : {item}, idx: {idx}, len : {len(sample['tokens'])}")
        tokens = sample["tokens"][start_length:]
        pack_tokens.extend(tokens)
        pack_labels.extend(tokens[1:] + [-100])
        type_ids.extend([sample["type_id"]] * len(tokens))

        for i in range(start_idx + 1, end_idx):
            idx = self.indices[i]
            sample = self.dataset[idx]
            self.pdebug(f"item : {item}, idx: {idx}, len : {len(sample['tokens'])}")
            tokens = sample["tokens"]
            pack_tokens.extend(tokens)
            pack_labels.extend(tokens[1:] + [-100])
            type_ids.extend([sample.get("type_id")] * len(tokens))

        # corner case, the last sample is useless
        if end_length == 0:
            pass
        else:
            idx = self.indices[end_idx]
            sample = self.dataset[idx]
            self.pdebug(f"item : {item}, idx: {idx}, len : {len(sample['tokens'])}")
            tokens = sample["tokens"][:end_length]
            pack_tokens.extend(tokens)
            pack_labels.extend(tokens[1:] + [-100])
            type_ids.extend([sample.get("type_id")] * len(tokens))

        return {
            "tokens": pack_tokens,
            "cu_seqlens": [i * self.max_length_per_sample for i in range(self.bsz + 1)],
            "indexes": list(range(self.max_length_per_sample)) * self.bsz,
            "labels": pack_labels,
            "type_ids": type_ids,
        }


def get_packed_dataset_without_short_length(
    folder,
    max_length_per_sample=2048,
    packed_length=4096,
    show_progress=False,
    min_length=50,
    min_length_dict=None,
    pack_into_one_sample=False,
):
    """
    Given a folder, combine all the .bin files into a single large dataset.
    And filter out short samples with length less than 'min_length'.

    Each .bin file is treated as a separate dataset.

    Args:
        folder (str): Path to the folder containing the .bin files.
        max_length_per_sample (int): Maximum length of each sample.
        packed_length (int): Length to pack samples to.
        show_progress (bool): Whether to show the progress bar.
        min_length (int): The minimum length of the sample.
        min_length_dict (dict): The minimum length of the sample for each dataset.
         The format is something like {'pile-arxiv': 50}
        dataset_backend (Optional[str]): Dataset storage location. Optional parameters are local, local-shm, kv

    Returns:
        A packed dataset containing all the data from the .bin files.
    """

    assert os.path.exists(folder), f"{folder} does not exist."
    datasets = []
    delete_samples = 0

    for root, dirs, files in os.walk(folder, followlinks=True):
        dirs.sort()  # Let the folder need to be returned in a fixed order
        if gpc.is_rank_for_log():
            logger.info(f"Reading {root}...")
        num_token_in_folder = 0

        for fn in tqdm(sorted(files), total=len(files), leave=False, disable=not show_progress):
            if fn.endswith(".bin"):
                fp = os.path.join(root, fn)
                catch_ml_keys = []
                min_length_num = min_length
                if min_length_dict is not None:
                    for k, v in min_length_dict.items():
                        if k in fp:
                            min_length_num = v
                            catch_ml_keys.append(k)
                    assert (
                        len(catch_ml_keys) < 2
                    ), f"The file name `{fp}` matched the following resample keys:{catch_ml_keys}"

                ds_type_id = get_dataset_type_id(path=fp)
                ds = JsonlDataset(fp, ds_type_id, min_length=min_length_num)

                if hasattr(ds, "old_length"):
                    delete_samples += ds.old_length - len(ds)
                if len(ds) == 0:
                    if gpc.is_rank_for_log():
                        logger.info(f"None of the data in `{fp}` is longer than {min_length}")
                    continue

                if pack_into_one_sample:
                    ds = PackedDatasetWithoutCuSeqlen(ds, max_length_per_sample, packed_length)
                else:
                    ds = PackedDataset(ds, max_length_per_sample, packed_length)

                num_token_in_folder += len(ds) * packed_length
                datasets.append(ds)

    dataset = ConcatDataset(datasets=datasets)
    if gpc.is_rank_for_log():
        logger.info(
            f"Find `{len(datasets)}` datasets, \
            {len(dataset)} samples, \
            delete `{delete_samples}` because of short length",
        )

    return dataset
