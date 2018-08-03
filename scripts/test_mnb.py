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


    
def test_with_mnb(model, data, task, criterion, cuda, bs, mean, std, logger):
    """Tests or validates model using minibatches"""
    
    n = len(data)
    dual = model.dual
    J = model.J
    logging.warning('Testing on {} molecules'.format(n))

    loss = utils.AverageMeter()
    error = utils.AverageMeter()
    
    batch_idx = batching.get_batches(n,bs,data,False,False)
    
    #criterion = nn.MSELoss()
    model.eval()
    
    t0 = time.time()
    for i,b in enumerate(batch_idx):
        
        batch = [data[j] for j in b]
        bsi = len(batch)
        
        X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch = batching.prepare_batch(batch, task, J)
        
        if mean==0: #generated data
            T = (T.squeeze()).type(torch.LongTensor)
        else:
            T = utils.normalize_data(T, mean, std)
        
        if cuda == True:
            X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch =  X.cuda(), W.cuda(), T.cuda(), XL.cuda(), WL.cuda(), Pm.cuda(), Pd.cuda(), mask.cuda() , mask_lg.cuda(), N_batch.cuda(), E_batch.cuda()  
        
        if dual == False:
            output = model([X, W], N_batch, mask)
                
        else :
            output = model([X, XL, W, WL, Pm, Pd], N_batch, mask, E_batch, mask_lg)
         
        #print(output, T)
        test_loss = criterion(output, T).item()
        dur = int(time.time() - t0)
        
        loss.update(test_loss,bsi)
        
        if mean!=0:
            error.update(utils.evaluation(output, T).item(),bsi)
        
        """
        logger.add_batch_info(i+1,loss.val, error.val, dur)
        logging.info("Batch {} Loss : {:.3f} Error : {:.3f}".format(i+1, test_loss, error.val))
        """
        
        t0 = time.time()
        
        """
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
        """
    return loss.avg, error.avg

  




