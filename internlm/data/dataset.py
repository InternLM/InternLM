import os
from typing import Dict

from torch.utils.data import ConcatDataset

from internlm.data.single_dataset import JsonlDataset


def get_dataset_dict(folder, split="valid") -> Dict:
    """
    Return a dictionary of Datasets from a folder containing data files for validation.

    Args:
        folder (str): The path to the folder containing data files.
        split (str): The split of the data files to be used, default is "valid".

    Returns:
        A dictionary containing Datasets for each folder in the given path
        that contains data files with the specified split.

    Raises:
        AssertionError: If the given folder does not exist.

    Example:
        If the given folder is as follows,
        - data
            - zhihu
                - xxx.bin
                - valid.bin
            - baike
                - xxx.bin
                - valid.bin

        The returned dictionary will be,
        {
            'zhihu': Dataset,
            'baike': Dataset
        }
    """

    assert os.path.exists(folder), f"folder `{folder}` not exists"
    data_dict = {}

    for root, dirs, files in os.walk(folder, followlinks=True):
        dirs.sort()  # The order is guaranteed, and the newly added data starting with z needs to be ranked behind
        datasets = []
        for fn in sorted(files):  # Need sorted to ensure that the order is consistent
            if fn.endswith(".bin") and split in fn:
                fp = os.path.join(root, fn)
                ds = JsonlDataset(fp)
                datasets.append(ds)
        if datasets:
            ds = ConcatDataset(datasets=datasets)
            data_dict[os.path.basename(root)] = ds

    return data_dict
