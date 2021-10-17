#! /usr/bin/env python3
import torch.nn as nn
import torch

class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class nnUnsqueeze(nn.Module):
    def __init__(self):
        super(nnUnsqueeze, self).__init__()

    def forward(self, x):
        return x[:, :, None, None]

class nnClamp(nn.Module):
    def __init__(self, min, max):
        super(nnClamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class nnReshape(nn.Module):
    def __init__(self, shape):
        super(nnReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)



"""
args:
    layers: list of dicts with fields
    {
        type: "type",
        <args>
    }
"""

def str_to_class(classname:str):
    if classname == "Conv1d": return nn.Conv1d
    if classname == "ReLU": return nn.ReLU
    if classname == "Flatten": return nn.Flatten
    if classname == "Linear": return nn.Linear
    if classname == "BatchNorm1d": return nn.BatchNorm1d
    if classname == "Dropout": return nn.Dropout
    if classname == "Conv2d": return nn.Conv2d
    if classname == "BatchNorm2d": return nn.BatchNorm2d
    if classname == "nnSqueeze": return nnSqueeze
    if classname == "nnClamp": return nnClamp
    if classname == "Identity": return nn.Identity
    if classname == "nnReshape": return nnReshape

    raise Exception("classname {} not found".format(classname))

def build_sequential(layers:list)->nn.Sequential:
    seq = []
    for l in layers:
        seq.append(str_to_class(l["class"])(**l["kargs"]))
    
    return nn.Sequential(*seq)

###############################################################################

nn.Sequential(
    nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1, padding=0),
    nn.ReLU(inplace=True), # Optionally do the operation in place,
    nn.Flatten(),
    nn.Linear(in_features=10700+100,out_features=1000), 
    nn.BatchNorm1d(num_features=123),
    nn.Dropout(p=0.5),
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
    nn.BatchNorm2d(num_features=1023), nn.ReLU(True),
    nnSqueeze(),
    nnClamp(0, 1),
    nn.Identity(),    
    nnReshape([-1])
)

###############################################################################

if __name__ == "__main__":
    seq = build_sequential(
        [
            {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
            {"class": "ReLU", "kargs": {"inplace": True}},
            {"class": "Flatten", "kargs": {}},
            {"class": "Linear", "kargs": {"in_features": 10700+100, "out_features": 1000}},
            {"class": "BatchNorm1d", "kargs": {"num_features": 123}},
            {"class": "Dropout", "kargs": {"p": 0.5}},
            {"class": "Conv2d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 }},
            {"class": "BatchNorm2d", "kargs": {"num_features":1023}},
            {"class": "nnSqueeze", "kargs": {}},
            {"class": "nnClamp", "kargs": {"min": 0, "max": 1}},
            {"class": "Identity", "kargs": {}},
            {"class": "nnReshape", "kargs": {"shape":[-1]}},
        ]
    )

    print(seq)

    # print(list(seq.parameters()))
