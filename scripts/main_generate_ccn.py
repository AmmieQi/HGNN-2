#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:49:32 2018

@author: sulem
"""
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

import sys
sys.path.insert(0, '/misc/vlgscratch4/BrunaGroup/sulem/chem/HGNN')

from models.compnets import model_ccn
#from preprocessing import preprocessing
from functions import utils, logs, utils_ccn
from scripts import train_ccn, test_ccn

# Logging settings
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
"""
handler = logging.FileHandler('logging.log')
log.addHandler(handler)
"""

# Argument parser
global parser
parser = argparse.ArgumentParser(description='GNN on QM9 dataset')

add_arg = parser.add_argument  

add_arg('--train_path', dest = 'train_path', 
        default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/generated/cp_train_4800.pickle')
add_arg('--valid_path', dest = 'valid_path',
        default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/generated/cp_valid_600.pickle')
add_arg('--test_path', dest = 'test_path',
        default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/generated/cp_test_600.pickle')
add_arg('--log_path', dest = 'log_path', default=None)

add_arg('--train', dest='train', help='training',default=True)
add_arg('--val', dest='val', help='validate',default=True)
add_arg('--test', dest='test', help='testing',default=False)

# Optimization options
add_arg('--epochs', dest='max_epoch', help='num epochs', default=20,type=int)
add_arg('--step', dest='epoch_step', default=5,type=int)

add_arg('--optim', dest='optim', help='Optimization algorithm', type=str, default='adamax')
add_arg('--lr', dest='lr', help='learning rate', type=float, default=0.001)
add_arg('--lrdamping', dest='lrdamping', help='learning rate damping',
        type=float, default=0.85)
add_arg('--momentum', dest='momentum', default=0.9,type=float)

# Model options
add_arg('--model', dest = 'model_path', default=None)
add_arg('--k', dest = 'order', help="Model order", default=1,type=int)
add_arg('--cuda',dest='cuda', help='Enables CUDA', default=True,type=bool)
add_arg('--L',dest='layers', help='input layers', default=10,type=int)
add_arg('--h', dest='hidden_size', help='feature maps', default=2,type=int)
add_arg('--task', dest='task', default=0,type=int)

    
def main():
    
    global args
    args = parser.parse_args()
    if args.log_path == None:
        log_path = ('log/simul_data/ccn_' + str(args.order) + '_ep_' + str(args.max_epoch) + '_st_' + str(args.epoch_step)
                    + '_op_' + str(args.optim) + '_lr_' + str(args.lr) + '_da_' + str(args.lrdamping)
                    + '_L_' + str(args.layers) + '_h_' + str(args.hidden_size) + '_ta_' +
                    str(args.task)  + '_' + str(time.time())[-3:] + '.pickle'
        )
        args.log_path = log_path
    log.info("Log path : " + log_path)
    
    # logger
    logger = logs.Logger(args.log_path)
    logger.write_settings(args)
    
    # Check if CUDA is enabled
    if args.cuda== True and torch.cuda.is_available():
        log.info('Working on GPU')
        #torch.cuda.manual_seed(0)
        
    else:
        log.info('Working on CPU')
        args.cuda = False
        #torch.manual_seed(0)

    # load training, validation and test datasets
    if args.train==True:
        with open(args.train_path,'rb') as file :
            train_set = pickle.load(file)
            Ntrain = len(train_set)   
            log.info("Number of training instances : " + str(Ntrain))
            logger.add_info('Training set size : ' + str(Ntrain))
    if args.val==True:
        with open(args.valid_path,'rb') as file :
            valid_set = pickle.load(file)
            Nvalid = len(valid_set) 
            log.info("Number of validation instances : " + str(Nvalid))
            logger.add_info('Validation set size : ' + str(Nvalid))
    if args.test==True:
        with open(args.test_path,'rb') as file :
            test_set = pickle.load(file)
            Ntest = len(test_set) 
            log.info("Number of test instances : " + str(Ntest))
            logger.add_info('Test set size : ' + str(Ntest))
    
    dim_input = train_set[0][0].size()[1]
    logger.add_info('Number of features of the inputs : ' + str(dim_input))

    # Creates or loads model
    if args.train == False or args.model_path != None:
        ccn = torch.load(args.model_path)
        log.info('Network loaded')
    else:
        if args.order==1 :
            ccn = model_ccn.CCN_1D(dim_input, 2, args.hidden_size,args.layers, args.cuda)
            logger.add_model('first order CCN')
            log.info('First-order CCN created')
            
        elif args.order==2:
            ccn = model_ccn.CCN_2D(dim_input, 2, args.hidden_size, args.layers, args.cuda)
            logger.add_model('second order CCN')
            log.info('Second-order CCN created')
        else:
            log.info('Order not implemented yet, second-order CNN will be created')
            ccn = model_ccn.CCN_2D(dim_input, 2, args.hidden_size, args.layers, args.cuda)
    
    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(ccn.parameters(), lr=args.lr,
                                       momentum=args.momentum)
    elif args.optim == 'adamax':
        optimizer = torch.optim.Adamax(ccn.parameters(), lr=args.lr)
    
    else :
        optimizer = torch.optim.Adam(ccn.parameters(), lr=args.lr)
        
    if args.cuda == True :    
        ccn = ccn.cuda()
    
    # Training
    
    if args.train==True:
        ccn.train()
        
        log.info('Training the CCN')
        logger.add_res('Training phase')
        
        run_loss = utils.RunningAverage()
        #run_error = utils.RunningAverage()
        
        for epoch in range (args.max_epoch):
            
            t0 = time.time()
            
            if epoch != 0 and epoch % args.epoch_step == 0 :
                args.lr = args.lr * args.lrdamping
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            
            loss, _= train_ccn.train_ccn(ccn, train_set, args.task, criterion,
                                         optimizer, args.cuda, 0, 1)
            
            dur = int(time.time() - t0)
            
            run_loss.update(loss)
            #run_error.update(error)
            
            logger.add_epoch_info(epoch+1,run_loss.val, run_loss.val, dur)
            log.info('Epoch {} : Average Loss {:.3f} Time : {}'
              .format(epoch+1, run_loss.val, dur))
        
        training_time = sum(logger.time_epoch)
        #ratio = run_error.val
        
        logger.add_train_info(run_loss.val, run_loss.val, training_time,run_loss.val)    
        log.info('Training finished : Duration {} secs, Avg Loss {:.3f}'
              .format(training_time, run_loss.val))
        
        logger.save_model(ccn)
    
    
    # Validating
    
    if args.val==True:
        log.info('Evaluating on the validation set...')
        logger.add_res('Validation phase')
        val_loss, _, dur = test_ccn.test_ccn(ccn, valid_set, args.task, criterion,
                                                       args.cuda, 0, 1, logger)
        #ratio_val = val_error
        log.info('Validation finished : Avg loss {:.3f} Duration: {} seconds'
                 .format(val_loss, dur))
        logger.add_test_perf(val_loss, val_loss,val_loss)
        
        logger.plot_train_logs()
        #logger.plot_test_logs()    
        
    
    if args.test==True:
        log.info('Evaluating on the test set...')
        logger.add_res('Test phase')
        test_loss, _ = test_ccn.test_ccn(ccn, test_set, args.task, criterion,
                                                       args.cuda, 0, 1, logger)
        #ratio_test = test_error
        log.info('Test finished : Avg loss {:.3f} Duration: {} seconds'
                 .format(test_loss, dur))
        logger.add_test_perf(test_loss, test_loss,test_loss)
        
        logger.plot_train_logs()
        #logger.plot_test_logs()    
        
        return test_loss

if __name__ == '__main__':
    main()
  




