import torch
import math
 
def split_dataset_by_percentage(train:float, val:float, test:float, dataset, seed:int):
    assert train < 1.0
    assert val < 1.0
    assert test < 1.0
    assert train + val + test <= 1.0

    num_train = math.floor(len(dataset) * train)
    num_val   = math.floor(len(dataset) * val)
    num_test  = math.floor(len(dataset) * test)

    return torch.utils.data.random_split(dataset, (num_train, num_val, num_test), generator=torch.Generator().manual_seed(seed))
