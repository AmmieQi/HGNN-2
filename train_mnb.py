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
logging.basicConfig(level=logging.DEBUG)

from model import *
from preprocessing import *
from logger import *
import batching 
import utils


def train_with_mnb(model, data, task, criterion, optimizer, cuda, bs, mean, std):
    """Trains model for one epoch using minibatches"""
    
    n = len(data)
    dual = model.dual
    J = model.J
    #logging.warning('Training on {} molecules'.format(n))
    
    load_time = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    error_ratio = utils.AverageMeter()
    
    criterion = nn.MSELoss()
    model.train()    
    
    batch_idx = batching.get_batches(n,bs,data,False,False)
    
    t0 = time.time()
    for i, b in enumerate(batch_idx):
        
        optimizer.zero_grad()
        
        batch = [data[j] for j in b]
        bsi = len(batch)
        
        X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch = batching.prepare_batch(batch, task, J)
        T = utils.normalize_data(T, mean, std)
        
        if cuda == True:
            X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch =  X.cuda(), W.cuda(), T.cuda(), XL.cuda(), WL.cuda(), Pm.cuda(), Pd.cuda(), mask.cuda() , mask_lg.cuda(), N_batch.cuda(), E_batch.cuda()  
            criterion = criterion.cuda()
        
        load_time.update(int(time.time() - t0))
        
        if dual == False:
            output = model([X, W], N_batch, mask)
                
        else :
            output = model([X, XL, W, WL, Pm, Pd], N_batch, mask, E_batch, mask_lg)
            
        train_loss = criterion(output, T)
        # Logs
        losses.update(train_loss.item(), bsi)
        error_ratio.update(evaluation(output, T).item(), bsi)
        
        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()
        
        batch_time.update(int(time.time() - t0))
        t0 = time.time()
        
        
        if i < 10 or ( i < 200 and i % 20 == 0) :
            
            print('Batch {} : \t'
                  #'Loading time {batch_time.avg:.3f} \t'
                  #'Preprocessing time {data_time.avg:.3f} \t'
                  'Loss : {loss.avg:.4f} \t'
                  'Error Ratio : {err.val:.4f}'
                  .format(i, batch_time=batch_time,data_time=load_time, loss=losses, err=error_ratio))
        if i % 20 == 0 :    
            print(output)
            print(T)
        
    
    return losses.avg, error_ratio.avg
  




