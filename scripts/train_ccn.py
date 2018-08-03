#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:49:32 2018

@author: sulem
"""

import time

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import logging

#from models.compnets.model_ccn import CCN_1D
from functions import utils

    
def train_ccn(net, data, task, criterion, optimizer, cuda, mean, std):
    """Trains model for one epoch"""
    
    n = len(data)
    losses = utils.RunningAverage()
    error = utils.RunningAverage()
    
    for i in range (n):
        
        optimizer.zero_grad()
            
        X, A, targets,_,_,_,_ = data[i]
        A = A + torch.eye(A.shape[0])
        y = targets[task].view(1)
        
        if mean==0: #generated data
            y = y.type(torch.LongTensor)
        else:
            y = (y - mean) / (std + 10 ** -8)
        
        X.requires_grad = True
        A.requires_grad = True
        
        if cuda :
            X = X.cuda()
            A = A.cuda()            
            y = y.cuda()
        
        output = net(X,A)
        
        if mean!=0:
            error.update(utils.evaluation(output, y).item())
        else:
            output = output.view(1,-1)
        
        #print(y.shape)
        #print(output.shape)
        loss = criterion(output, y)
        losses.update(loss.item())
        
        """
        if i % 400 == 0 :
            logging.info('After {} instances : Loss {:.3f} MAE {:.3f}'.format(i,losses.val,error.val)) 
        """
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
    return losses.val, error.val

  




