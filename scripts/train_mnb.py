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
logging.basicConfig(level=logging.DEBUG)

from functions import batching, utils, logs

# device = "gpu:0" if torch.cuda.is_available() else "cpu"

def train_with_mnb(model, data, task, criterion, optimizer, cuda, bs, mean, std):
    """Trains model for one epoch using minibatches"""
    
    n = len(data)
    dual = model.dual
    J = model.J
    #logging.warning('Training on {} molecules'.format(n))
    
    losses = utils.RunningAverage()
    error = utils.RunningAverage()

    model.train()    
    
    batch_idx = batching.get_batches(n,bs,data,False,False)
    #print(batch_idx)
    
    for i, b in enumerate(batch_idx):
        
        optimizer.zero_grad()
        
        batch = [data[j] for j in b]
        bsi = len(batch)
        
        X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch = batching.prepare_batch(batch, task, J)
        
        if mean==0: #generated data
            T = (T.squeeze()).type(torch.LongTensor)
        else:
            T = utils.normalize_data(T, mean, std)
        #print("Batch {} prepared".format(i+1))
        
        X.requires_grad = True
        W.requires_grad = True
        
        if cuda == True:
            X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch =  X.cuda(), W.cuda(), T.cuda(), XL.cuda(), WL.cuda(), Pm.cuda(), Pd.cuda(), mask.cuda() , mask_lg.cuda(), N_batch.cuda(), E_batch.cuda()  
        
        if dual == False:
            XL.requires_grad = True
            WL.requires_grad = True
            Pm.requires_grad = True
            Pd.requires_grad = True
            output = model([X, W], N_batch, mask)
                
        else :
            output = model([X, XL, W, WL, Pm, Pd], N_batch, mask, E_batch, mask_lg)
            
        #print(output.shape)
        #print(output, T)
        #print(T.shape)

        train_loss = criterion(output, T)
        #logging.info("Training loss : {:.3f}".format(train_loss))
        # Logs
        losses.update(train_loss.item())
        
        if mean!=0:
            error.update(utils.evaluation(output, T).item())
        
        """
        if i % 20 == 0 :
            logging.info('Batch {} : Loss {:.3f} MAE {:.3f}'.format(i+1,losses.val,error.val)) 
        """
        
        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()
        
    
    return losses.val, error.val
  




