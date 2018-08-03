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

from models.layers import layers_mnb
    

class GNN_simple(nn.Module):
    
    """Power GNN (article Community Detection with Hierarchical GNN)
    
        Parameters :
        
            task : int between 0 and 13
            
            n_features (number of features in the first layer) : int
            
            n_layers (number of iterations) : int
            
            dim_input : int
            
            J : int
            
    """
    
    def __init__(self, task, n_features, n_layers, dim_input, dim_output=1, J=1, gru=False):
        super(GNN_simple, self).__init__()
        
        self.dual = False
        self.J = J
        self.gru = False
        
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_outputs = dim_output

        self.featuremap_in = [dim_input, n_features]
        self.featuremap_mi = [2*n_features, n_features]
        self.featuremap_end = [2*n_features, dim_output]
        
        self.layer0 = layers_mnb.layer_simple(self.featuremap_in, J+2, gru)
        for i in range(n_layers-2):
            module = layers_mnb.layer_simple(self.featuremap_mi, J+2, gru)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = layers_mnb.layer_last(self.featuremap_end, J+2)

    def forward(self, state, N_batch, mask):
        cur = self.layer0(state, N_batch, mask)
        #print(cur[0])
        for i in range(self.n_layers-2):
            cur = self._modules['layer{}'.format(i+1)](cur, N_batch, mask)
            #print(cur[0])
        out = self.layerlast(cur, N_batch, mask)
        #print(out[0])
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
    
    def __init__(self, task, n_features, n_layers, dim_input, dim_output=1, J=1, order = 1):
        super(GNN_lg, self).__init__()
        
        self.dual = True
        self.J = J
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_outputs = dim_output
        self.order = order
        
        self.featuremap_in = [dim_input, 1, n_features]
        self.featuremap_mi = [2*n_features, 2*n_features, n_features]
        self.featuremap_end = [2*n_features, dim_output]
        
        if order == 1 :
            self.layer0 = layers_mnb.layer_with_lg_1(self.featuremap_in, J+2)
            for i in range(n_layers-2):
                module = layers_mnb.layer_with_lg_1(self.featuremap_mi, J+2)
                self.add_module('layer{}'.format(i + 1), module)
        
        elif order == 2 :
            self.layer0 = layers_mnb.layer_with_lg_2(self.featuremap_in, J+2)
            for i in range(n_layers-2):
                module = layers_mnb.layer_with_lg_2(self.featuremap_mi, J+2)
                self.add_module('layer{}'.format(i + 1), module)


        else :
            self.layer0 = layers_mnb.layer_with_lg_3(self.featuremap_in, J+2)
            for i in range(n_layers-2):
                module = layers_mnb.layer_with_lg_3(self.featuremap_mi, J+2)
                self.add_module('layer{}'.format(i + 1), module)
                
        self.layerlast = layers_mnb.layer_last_lg(self.featuremap_end, J+2)
        
        
    def forward(self, state, N_batch, mask, E_batch, mask_lg):
        cur = self.layer0(state, N_batch, mask, E_batch, mask_lg)
        for i in range(self.n_layers-2):
            cur = self._modules['layer{}'.format(i+1)](cur, N_batch, mask, E_batch, mask_lg)
        out = self.layerlast(cur, N_batch, mask)
        return out