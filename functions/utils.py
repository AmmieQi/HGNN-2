#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:19:00 2018

@author: sulem
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

cuda = True

if torch.cuda.is_available() and cuda == True:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

    
def graph_op(A,X):
    
    """ Tensor multiplication and resizing for the convolution layers :
        
        tensor A (bs x N x N x J) contains J graph operators of bs graphs
        
        tensor X (bs x n_features x N)
        
        output : (bs x J*n_features x N)
    """
    
    bs = A.size()[0]
    N = A.size()[1]
    J = A.size()[3]
    n_feat = X.size()[1]
    
    output = torch.zeros(bs, J*n_feat, N).type(dtype) 
    
    for b in range (bs) :
        
        for j in range (J) :
            
            Aslice =  A[b,:,:,j] # N x N
            Xslice = X[b,:,:].transpose(1,0) # N x n_feat
            AX = torch.mm(Aslice, Xslice) # dimension N x n_feat
            
            output[b,n_feat*j:n_feat*(j+1),:] = AX.transpose(1,0)
 
    return output


def Pmul(P,X) :
    
    """ Tensor multiplication and resizing :
    
    tensor P (bs x N x M)
    
    tensor X (bs x n_features x M)
    
    output : (bs x n_features x N)
    """

    bs = P.size()[0]
    N = P.size()[1]
    #M = P.size()[2]
    n_feat = X.size()[1]
    
    output = torch.zeros(bs, n_feat, N).type(dtype) 
    
    for b in range (bs) :
        
        Pslice =  P[b,:,:] # N x M
        Xslice = X[b,:,:].transpose(1,0) # M x n_feat
        AX = torch.mm(Pslice, Xslice) # dimension N x n_feat
        
        output[b,:,:] = AX.transpose(1,0)
 
    return output


def normalize_data(data, mean=None, std=None):
    
    if mean==None or std==None :
        _, _, mean , std = data_stats(data)
    
    if std < 10 ** -5 :
        data_norm = data - mean
    
    else:
        data_norm = (data - mean) / std
    
    return data_norm


def evaluation(pred, target):
    
    err = torch.mean(torch.abs(pred - target))
    
    return err


def data_stats(data):
    
    minimum = torch.min(data)
    maximum = torch.max(data)
    mean = torch.mean(data)
    std = 10 ** -5 + torch.std(data)
    
    return minimum.item(), maximum.item(), mean.item(), std.item()


class AverageMeter():
    
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class RunningAverage():
    
    """ Computes a running average with momentum """
    
    def __init__(self, momentum=0.1):
        self.momentum = momentum
        self.val = 0.0

    def update(self, val):
        if self.val == 0.0 :
            self.val = val
        else:
            self.val = (1 - self.momentum) * val + self.momentum * self.val