#!/usr/bin/env python
# -*- encoding: utf-8 -*-

DATASET_TYPE_IDS_MAP = {"en": 0, "cn": 1, "code": 2, "ja": 3, "ar": 4, "kaoshi": 5}


def get_dataset_type_id(path):
    import re

    match_idxes = []
    for key, idx in DATASET_TYPE_IDS_MAP.items():
        if re.search(rf"/[z_]*{key}/", path):
            match_idxes.append(idx)
    assert len(match_idxes) == 1, f"{path}, match_idxes should be 1, but got {match_idxes} from {DATASET_TYPE_IDS_MAP}"
    return match_idxes[0]
