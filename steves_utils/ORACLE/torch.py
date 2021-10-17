#! /usr/bin/env python3

import math
import torch

from steves_utils.ORACLE.ORACLE_sequence import ORACLE_Sequence
from steves_utils.lazy_map import Lazy_Map

class ORACLE_Torch_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        desired_serial_numbers,
        desired_runs,
        desired_distances,
        window_length,
        window_stride,
        num_examples_per_device,
        seed,
        max_cache_size=1e6,
        transform_func=None,
        prime_cache=False,
    ) -> None:
        super().__init__()

        self.os = ORACLE_Sequence(
            desired_serial_numbers,
            desired_runs,
            desired_distances,
            window_length,
            window_stride,
            num_examples_per_device,
            seed,
            max_cache_size,
            prime_cache=prime_cache
        )

        self.transform_func = transform_func

    def __len__(self):
        return len(self.os)
    
    def __getitem__(self, idx):
        if self.transform_func != None:
            return self.transform_func(self.os[idx])
        else:
            return self.os[idx]
    
def split_dataset_by_percentage(train:float, val:float, test:float, dataset, seed:int):
    assert train < 1.0
    assert val < 1.0
    assert test < 1.0
    assert train + val + test <= 1.0

    num_train = math.floor(len(dataset) * train)
    num_val   = math.floor(len(dataset) * val)
    num_test  = math.floor(len(dataset) * test)

    return torch.utils.data.random_split(dataset, (num_train, num_val, num_test), generator=torch.Generator().manual_seed(seed))