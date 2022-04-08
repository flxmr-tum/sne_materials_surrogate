import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SimpleNetBN(nn.Module):
    def __init__(self, feature_dims : int = 1, bnlayers : int = 4, llayers : int = 4,
                 dropout_on=False, activation="softplus"):
        super().__init__()

        self.bnlayers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.llayers = nn.ModuleList()
        

        layer_sizes = list(np.linspace(feature_dims, 1, num=bnlayers+llayers, endpoint=False, dtype=np.int))

        if bnlayers:
            for i, o in zip(layer_sizes[:bnlayers+1][:-1], layer_sizes[:bnlayers+1][1:]):
                self.bnlayers.append(nn.Linear(i,o, bias=False))
                self.batchnorms.append(nn.BatchNorm1d(o))

        if llayers:
            for i, o in zip(layer_sizes[bnlayers:][:-1], layer_sizes[bnlayers:][1:]):
                self.llayers.append(nn.Linear(i,o, bias=False))

        self.regressor_layer = nn.Linear(layer_sizes[-1], 1)

        if activation == 'relu':
            self.actf = F.leaky_relu
        elif activation == 'tanh':
            self.actf = F.tanh
        elif activation == 'softplus':
            self.actf = F.softplus
        self.dropout_on = dropout_on
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        for fw, bn in zip(self.bnlayers, self.batchnorms):
            if self.dropout_on:
                x = self.actf(bn(self.drop(fw(x))))
            else:
                x = self.actf(bn(fw(x)))
        for fw in self.llayers:
            #x = F.tanh(fw(x))
            x = self.actf(fw(x))
        x = self.actf(self.regressor_layer(x))
        return x
    
class SimpleNetBN_TG(nn.Module):
    def __init__(self, feature_dims : dict = {"fps" : 1}, bnlayers : int = 4, llayers : int = 4):
        super().__init__()

        self._USE_TORCH_GEOMETRIC = True

        self.bnlayers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.llayers = nn.ModuleList()
        
        self.feature_dims = feature_dims["fps"]
        layer_sizes = list(np.linspace(self.feature_dims, 1, num=bnlayers+llayers, endpoint=False, dtype=np.int))

        if bnlayers:
            for i, o in zip(layer_sizes[:bnlayers+1][:-1], layer_sizes[:bnlayers+1][1:]):
                self.bnlayers.append(nn.Linear(i,o, bias=False))
                self.batchnorms.append(nn.BatchNorm1d(o))

        if llayers:
            for i, o in zip(layer_sizes[bnlayers:][:-1], layer_sizes[bnlayers:][1:]):
                self.llayers.append(nn.Linear(i,o, bias=False))

        self.regressor_layer = nn.Linear(layer_sizes[-1], 1)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, data):
        x = data["fps"].reshape(-1, self.feature_dims)
        for fw, bn in zip(self.bnlayers, self.batchnorms):
            x = F.relu(bn(fw(x)))
        for fw in self.llayers:
            x = F.relu(fw(x))
        x = F.relu(self.regressor_layer(x))
        return x
