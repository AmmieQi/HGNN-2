#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:34:22 2018

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
logging.basicConfig(level=logging.DEBUG)

from functions import utils


def test_ccn(ccn, data, task, criterion, cuda, mean, std, logger):

    n = len(data)
    #logging.warning('Testing on {} molecules'.format(n))
    
    losses = utils.AverageMeter()
    error = utils.AverageMeter()
    ccn.eval()
    
    t0 = time.time()
    
    for i in range (n):
            
        X, A, targets,_,_,_,_ = data[i]
        A = A + torch.eye(A.shape[0])
        y  = targets[task].view(1)
        
        if mean==0: #generated data
            y = y.type(torch.LongTensor)
        else:
            y = (y - mean) / (std + 10 ** -8)
            
        if cuda :
            X, A, y = X.cuda(), A.cuda(), y.cuda()
            
        output = ccn(X,A)
        
        if mean!=0:
            error.update(utils.evaluation(output, y).item(), 1)
        else:
            output = output.view(1,-1)
            
        loss = criterion(output, y)
        losses.update(loss.item(), 1)
        
        """
        if i % 20 == 0:
            logger.add_batch_info(i+1,losses.avg, error.avg, 0)
            logging.info("After {} instances: Average Loss : {:.3f} Average Error : {:.3f}"
                  .format(i+1, losses.avg, error.avg))
        """
        
    test_time = int(time.time() - t0)
    
    return losses.avg, error.avg, test_time

