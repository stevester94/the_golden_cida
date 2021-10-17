#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from steves_utils.rotated_mnist_dataset import Rotated_MNIST_DS

class CIDA_MNIST_DS(torch.utils.data.Dataset):
    @classmethod
    def get_default_domain_configs(cls):
        return [
            {
                "domain_index":0,
                "min_rotation_degrees":0,
                "max_rotation_degrees":10,
                "num_examples_in_domain":5000,
            },
            {
                "domain_index":1,
                "min_rotation_degrees":11,
                "max_rotation_degrees":20,
                "num_examples_in_domain":5000,
            },
            {
                "domain_index":2,
                "min_rotation_degrees":21,
                "max_rotation_degrees":30,
                "num_examples_in_domain":5000,
            },
            {
                "domain_index":3,
                "min_rotation_degrees":31,
                "max_rotation_degrees":40,
                "num_examples_in_domain":5000,
            },
            {
                "domain_index":4,
                "min_rotation_degrees":41,
                "max_rotation_degrees":75,
                "num_examples_in_domain":5000,
            },
            {
                "domain_index":5,
                "min_rotation_degrees":76,
                "max_rotation_degrees":90,
                "num_examples_in_domain":5000,
            },
            {
                "domain_index":6,
                "min_rotation_degrees":91,
                "max_rotation_degrees":130,
                "num_examples_in_domain":5000,
            },
            {
                "domain_index":7,
                "min_rotation_degrees": 131,
                "max_rotation_degrees":180,
                "num_examples_in_domain":5000,
            },
        ]

    def __init__(self, seed, domain_configs, root="/mnt/wd500GB/CSC500/") -> None:
        """
        args:
            domain_configs: {
                "domain_index":int,
                "min_rotation_degrees":float,
                "max_rotation_degrees":float,
                "num_examples_in_domain":int,
            }
        """
        super().__init__()



        self.rng = np.random.default_rng(seed)

        self.data = []

        """
        Generate ranges to get something like
        0,120
        120,240
        240,360
        """
        # domain_ranges = []
        # temp_domain_ranges = np.linspace(min_rotation_degrees, max_rotation_degrees, num_domains+1)
        # for i, _ in enumerate(temp_domain_ranges):
        #     if i == len(temp_domain_ranges)-1:
        #         continue
        #     domain_ranges.append(
        #         (temp_domain_ranges[i], temp_domain_ranges[i+1])
        #     )
        
        for domain in domain_configs:
            assert(domain["min_rotation_degrees"] >= 0)
            assert(domain["max_rotation_degrees"] <= 360)
            ds = Rotated_MNIST_DS(seed, domain["min_rotation_degrees"], domain["max_rotation_degrees"], root=root)

            for rando in self.rng.choice(len(ds), size=domain["num_examples_in_domain"], replace=False):
                example = ds[rando]
                self.data.append(
                    (example[0], example[1], domain["domain_index"])
                )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

        

if __name__ == "__main__":
    import random

    cida_mnist_ds = CIDA_MNIST_DS(1337, CIDA_MNIST_DS.get_default_domain_configs())

    l = list(cida_mnist_ds)
    random.shuffle(l)
    

    print(len(l))

    for x,y,t,source in l:

        figure, axis = plt.subplots(1, 1, figsize=(10,10))

        axis.imshow(x[0])
        axis.set_title(
            f'Label: {y}\n'
            f'Domain: {t:.0f}\n'
            f'Source: {source}'
        )

        plt.show()