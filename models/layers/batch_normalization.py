#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:15:18 2018

@author: sulem
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    #torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    #torch.manual_seed(0)
    
class BN(nn.Module):
    def __init__(self, n_features,scale=0.1):
        super(BN, self).__init__()
        self.weight = nn.Parameter(torch.tensor(n_features).type(torch.FloatTensor))
        self.bias = nn.Parameter(torch.tensor(n_features).type(torch.FloatTensor))
        torch.nn.init.normal_(self.weight,0,scale)
        torch.nn.init.normal_(self.bias,0,scale)
        self.running_mean = torch.zeros(n_features).type(dtype)
        self.running_std = torch.zeros(n_features).type(dtype)
        self.momentum = 0.1
        
    def forward(self, X, N_batch, mask):
        if self.training:
            H, X_mean, X_std = sb_normalization(X, N_batch, mask)
            self.running_mean = (1-self.momentum) * X_mean + self.momentum * self.running_mean
            self.running_std = (1-self.momentum) * X_std + self.momentum * self.running_std
        else:
            #print(self.running_mean)
            H, _, _ = sb_normalization(X, N_batch, mask, self.running_mean, self.running_std)
        
        return self.weight * H + self.bias
        
class spatial_batch_norm(nn.Module): 
    
    """ Spatial Batch Normalization Layer for the 1D convolution layer
        with padded tensors """
    
    def __init__(self, n_feat):
        super(spatial_batch_norm, self).__init__()
        
        self.n_feat = n_feat
        self.layer = torch.nn.Conv1d(n_feat, n_feat, 1)


    def forward(self, X, N_batch, mask, mean=None, std=None):
        """input : tensor X of size (bs, n_features, N) """
        
        X_norm, _, _ = sb_normalization(X, N_batch, mask, mean, std)
        y = self.layer(X_norm)
        return y
    

def sb_normalization(H, N_batch, mask, mean=None,std=None):
    
    H = mask_embedding(H, mask)

    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        mean = mean_with_padding(H, N_batch, mask)
        var = 10**-5 + mean_with_padding((H.transpose(2,1) - mean).transpose(2,1) ** 2, N_batch, mask)
        std = var ** 0.5
    
    #print(mean)
    #print(std)
    H = ((H.transpose(2,1) - mean)  / std).transpose(2,1)
    return H, mean, std
    
    
def mean_with_padding(tensor, N_batch, mask):
    """ Get mean of tensor (bs, n_features, Nmax)
    accounting for zero padding of batches
    
    output (bs, n_features) """
    
    #print(tensor.shape)
    tensor = mask_embedding(tensor, mask)
    somme = torch.sum(torch.sum(tensor, dim=2), dim=0)
    n = torch.sum(N_batch)
    #print(tensor.shape)
    #print(somme.shape)
    #print(n.shape)
    return somme / n.item()


def mask_embedding(H, mask):
    
    """ Apply mask (matrix of dimensions bs x Nmax) to embedding H (tensor of
        dimensions bs x n_feat x Nmax) to pad the values of added nodes"""
    
    bs = mask.shape[0]
    N = mask.shape[1]
    nb_feat = H.shape[1]
    
    temp = (mask[:,:,0].view(bs,1,N)).repeat(1,nb_feat,1)
    H = torch.mul(H,temp)
    
    return H
