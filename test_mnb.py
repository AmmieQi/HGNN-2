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
from utils import *
from logger import *
import batching

    
def test_with_mnb(model, data, task, cuda, bs, mean, std, logger):
    """Tests or validates model using minibatches"""
    
    t0 = time.time()
    
    n = len(data)
    dual = model.dual
    J = model.J
    logging.warning('Testing on {} molecules'.format(n))
    
    load_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()
    
    batch_idx = batching.get_batches(n,bs,data,True,True)
    
    criterion = nn.MSELoss()
    model.eval()
    
    t0 = time.time()
    for i,b in enumerate(batch_idx):
        
        batch = [data[j] for j in b]
        bsi = len(batch)
        
        X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch = batching.prepare_batch(batch, task, J)
        T = normalize_data(T,mean,std)
        
        if cuda == True:
            X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch =  X.cuda(), W.cuda(), T.cuda(), XL.cuda(), WL.cuda(), Pm.cuda(), Pd.cuda(), mask.cuda() , mask_lg.cuda(), N_batch.cuda(), E_batch.cuda()  
            criterion = criterion.cuda()
        
        load_time.update(int(time.time() - t0))
        
        if dual == False:
            output = model([X, W], N_batch, mask)
                
        else :
            output = model([X, XL, W, WL, Pm, Pd], N_batch, mask, E_batch, mask_lg)
            
        test_loss = criterion(output, T)
        error = evaluation(output, T).item()
        dur = int(time.time() - t0)
        
        # Logs
        losses.update(test_loss.item(), bsi)
        error_ratio.update(error, bsi)
        batch_time.update(dur)
        
        logger.add_test_loss(i+1,losses.avg)
        logger.add_test_error(i+1,error_ratio.avg)
        logger.add_test_time(i+1,dur)
        
        losses.update(test_loss.item(), bsi)
        error_ratio.update(error, bsi)
        batch_time.update(dur)
        
        t0 = time.time()
        
        if i < 10 or ( i < 200 and i % 20 == 0) :
            logging.debug('Batch {} : \t'
                  #'Time {batch_time.avg} \t'
                  #'Batching {data_time.avg} \t'
                  'Loss : {loss.avg:.4f} \t'
                  'Error Ratio : {err.avg:.4f}'
                  .format(i, batch_time=batch_time,data_time=load_time, loss=losses, err=error_ratio))
        
        if i % 20 == 0 :
            logging.debug("output : " + str(output))
            logging.debug("target : " + str(T))
    return losses.avg, error_ratio.avg

  




