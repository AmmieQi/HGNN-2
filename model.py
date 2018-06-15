#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:48:58 2018

@author: sulem
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import layers
import layers_mnb
from utils import *
    

class GNN_simple(nn.Module):
    
    """Power GNN (article Community Detection with Hierarchical GNN)
    
        Parameters :
        
            task : int between 0 and 13
            
            n_features (number of features in the first layer) : int
            
            n_layers (number of iterations) : int
            
            dim_input : int
            
            J : int
            
    """
    
    def __init__(self, task, n_features, n_layers, dim_input, J):
        super(GNN_simple, self).__init__()
        
        self.n_features = n_features
        self.n_layers = n_layers
        
        self.featuremap_in = [dim_input, n_features]
        self.featuremap_mi = [2*n_features, n_features]
        self.featuremap_end = [2*n_features, 1]
        
        self.layer0 = layers.layer_simple(self.featuremap_in, J+2)
        for i in range(n_layers):
            module = layers.layer_simple(self.featuremap_mi, J+2)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = layers.layer_last(self.featuremap_end, J+2)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.n_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        return out
    

class GNN_lg(nn.Module):
    """ GNN on line graph with NB operator (article Community Detection with Hierarchical GNN)
    
        Parameters :
        
            task : int between 0 and 13
            
            n_features (number of features in the first layer) : int
            
            n_layers (number of iterations) : int
            
            dim_input : int
            
            (the dimensions of input edge features is assumed to be one (degree or raw distance)
            
            J : int
            
    """
    
    def __init__(self, task, n_features, n_layers, dim_input, J):
        super(GNN_lg, self).__init__()
        
        self.n_features = n_features
        self.n_layers = n_layers
        
        self.featuremap_in = [dim_input, 1, n_features]
        self.featuremap_mi = [2*n_features, 2*n_features, n_features]
        self.featuremap_end = [2*n_features, 1]
        
        self.layer0 = layers.layer_with_lg(self.featuremap_in, J+2)
        for i in range(n_layers):
            module = layers.layer_with_lg(self.featuremap_mi, J+2)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = layers.layer_last_lg(self.featuremap_end, J+2)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.n_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        return out