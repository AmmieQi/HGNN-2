#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:49:32 2018

@author: sulem
"""
import numpy as np
import argparse
import os
import os.path as path
import time
import pickle

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import logging

from model import *
from preprocessing import *
from utils import *
from logger import *

    
def test_model(model, data, task, dual, J, cuda, logger):
    """Tests or validates model over a set of instances"""
    
    t0 = time.time()
    
    n = len(data)
    logging.warning('Testing on {} molecules'.format(n))
    
    losses = AverageMeter()
    error_ratio = AverageMeter()
    
    criterion = nn.MSELoss()
    
    model.eval()
    
    for i in range (n):
            
        X, A, targets = data[i]
        target = targets[task]
        
        if dual==False :
            W = graph_operators([X,A],J,False)
            
            if cuda == True:
                X, W, target =  X.cuda(), W.cuda(), target.cuda()
                criterion = criterion.cuda()
            
            # Resizing for the GNN : X must be (bs, n_feat, N)
            #                        W must be (bs, N, N, J+3)
            X = X.transpose(1,0)
            X = X.view(1,X.size()[0],X.size()[1])
            W = W.view(1,W.size()[0],W.size()[1],W.size()[2])
            output = model([X, W])
                
        else :
            W, WL, Pm, Pd = graph_operators([X,A],J,True)
            XL = torch.diag(WL[:,:,1])
            
            if cuda == True:
                X, XL, W, WL, Pm, Pd, target =  X.cuda(), XL.cuda(), W.cuda(), WL.cuda(), Pm.cuda(), Pd.cuda(), target.cuda()
            
            # Resizing for the GNN : X must be (bs, n_feat, N)
            #                        XL must be (bs, n_edges, M)
            #                        W must be (bs, N, N, J+3)
            #                        WL must be (bs, M, M, J+3)
            #                        Pm and Pd must be (bs, N, M)
            X = X.transpose(1,0)
            X = X.view(1,X.size()[0],X.size()[1])
            XL = XL.view(1,1,XL.size()[0])
            W = W.view(1,W.size()[0],W.size()[1],W.size()[2])
            WL = WL.view(1,WL.size()[0],WL.size()[1],WL.size()[2])
            Pm = Pm.view(1,Pm.size()[0],Pm.size()[1])
            Pd = Pd.view(1,Pd.size()[0],Pd.size()[1])
            output = model([X, XL, W, WL, Pm, Pd])
            
        loss = criterion(output, target)
        error = evaluation(output, target)
        
        # Logs
        losses.update(loss.item(), 1)
        error_ratio.update(evaluation(output, target).item(), 1)
        
        logger.add_avg_test_loss(losses.avg)
        logger.add_avg_test_error(error_ratio.avg)
        
    test_time = time.time() - t0
    
    return error_ratio.avg, test_time

  




