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

import sys
sys.path.insert(0, '/misc/vlgscratch4/BrunaGroup/sulem/chem/HGNN')

from models.gnns import model_mnb
#from preprocessing import preprocessing
from functions import utils, logs
from scripts import train_mnb, test_mnb

# Logging settings
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
"""
handler = logging.FileHandler('logging.log')
log.addHandler(handler)
"""

# Argument parser
global parser
parser = argparse.ArgumentParser(description='GNN on QM9 dataset with non linearities')

add_arg = parser.add_argument  

add_arg('--train_path', dest = 'train_path', default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/train_set_400.pickle')
add_arg('--valid_path', dest = 'valid_path', default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/valid_set_100.pickle')
add_arg('--logPath', dest = 'log_path', default='/misc/vlgscratch4/BrunaGroup/sulem/chem/HGNN/log/test1/')

# Optimization options
add_arg('--batchsize', dest='batch_size', help='mini-batch size',default=10)
add_arg('--maxepoch', dest='max_epoch', help='num epochs', default=50)
add_arg('--epochstep', dest='epoch_step', default=10)

add_arg('--lr', dest='lr', help='learning rate', type=float, default=0.01)
add_arg('--lrdamping', dest='lrdamping', help='learning rate damping', type=float, default=0.9)
add_arg('--momentum', dest='momentum', default=0.9)
add_arg('--lrdecay', dest='lrDecay', help='learning rate decay', default=0.6)
#add_arg('--L', dest='L', help='epoch size', default=10)

# Model options
add_arg('--cuda',dest='cuda', help='Enables CUDA', default=True)
add_arg('--layers',dest='layers', help='input layers', default=10)
add_arg('--nfeatures', dest='nfeatures', help='feature maps', default=10)
add_arg('--J', dest='J', default=1)
add_arg('--task', dest='task', default=0)

    
def main():
    
    chem_acc = torch.tensor([0.1, 0.05, 0.043, 0.043, 0.043, 0.043,
                         0.043, 0.1, 10.0, 1.2, 0.043, 0.043, 0.0012])
    
    global args
    args = parser.parse_args()
    
    accuracy = chem_acc[args.task]
    
    # logger
    logger = logs.Logger(args.log_path)
    # Write experiment settings
    logger.write_settings(args)
    logger.add_info('Chemical accuracy for task {} : {:.2f}'.format(args.task, accuracy))
    
    # Check if CUDA is enabled
    if args.cuda== True and torch.cuda.is_available():
        log.info('Working on GPU')
        dtype = torch.cuda.FloatTensor
        torch.cuda.manual_seed(0)
        
    else:
        log.info('Working on CPU')
        args.cuda = False
        dtype = torch.FloatTensor
        torch.manual_seed(0)

    # load training and validation datasets
    with open(args.train_path,'rb') as file :
        train_set = pickle.load(file)
        
    with open(args.valid_path,'rb') as file :
        valid_set = pickle.load(file)
    
    Ntrain = len(train_set)
    Nvalid = len(valid_set)
    logger.add_info('Training set size : ' + str(Ntrain))
    logger.add_info('Validation set size : ' + str(Nvalid))
    
    train_target = torch.zeros(Ntrain)
    valid_target = torch.zeros(Nvalid)
    
    for i in range (Ntrain):
        train_target[i] = train_set[i][2][args.task]
    t_stats = utils.data_stats(train_target)
    mean = t_stats[2]
    std = t_stats[3]
    
    for i in range (Nvalid):
        valid_target[i] = valid_set[i][2][args.task]
    v_stats = utils.data_stats(valid_target)
    
    logger.add_info('Stats on the training set task [min, max, mean, std] : ' + str(t_stats))
    logger.add_info('Stats on the validation set task [min, max, mean, std] : ' + str(v_stats))
    
    dim_input = train_set[0][0].size()[1]

    # Creates 2 GNNs : 1 not using the line graph, 1 using it       
    gnn = model_mnb.GNN_simple(args.task, args.nfeatures, args.layers, dim_input,args.J)
    logger.add_model('gnn simple')
    gnn_lg = model_mnb.GNN_lg(args.task, args.nfeatures, args.layers, dim_input, args.J)
    logger.add_model('gnn with LG')
    log.info('2 networks created')
    
    # Criterion
    criterion = nn.MSELoss()
    # Optimizers
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    optimizer_lg = torch.optim.Adamax(gnn_lg.parameters(), lr=args.lr)
    
    if args.cuda == True :    
        gnn = gnn.cuda()
        gnn_lg = gnn_lg.cuda()
    
    # Training the 2 models successively
    
    gnn.train()
    log.info('Training the GNN without line graph')
    logger.add_res('Training the GNN without line graph')
    for epoch in range (args.max_epoch):
        
        t0 = time.time()
        
        if epoch != 0 and epoch % args.epoch_step == 0 :
            args.lr = args.lr * args.lrdamping
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
        loss, error = train_mnb.train_with_mnb(gnn, train_set, args.task, criterion,
                                     optimizer, args.cuda, args.batch_size, mean, std)
        
        dur = int(time.time() - t0)

        logger.add_train_loss(epoch+1,loss)
        logger.add_train_error(epoch+1,error)
        logger.add_epoch_time(epoch+1,dur)

        log.info('Epoch {} : Avg Error Ratio {:.3f}; Average Loss {:.3f} Time : {}'
          .format(epoch+1, error, loss, dur))
    
    avg_train_loss = np.mean(np.array(logger.loss_train))
    avg_train_error = np.mean(np.array(logger.error_train))
    training_time = sum(logger.time_epoch)
    MAE = avg_train_error * accuracy
    
    logger.add_res('Average training loss : {:.3f}'.format(avg_train_loss))
    logger.add_res('Average training error : {:.3f}'.format(avg_train_error))
    logger.add_res('Mean Absolute error : {:.3f}'.format(MAE))
    logger.add_res('Training time : {} seconds'.format(training_time))
    
    log.info('Avg Error Ratio {err:.3f}; Mean Average Error {MAE:.3f}'
          .format(err=avg_train_error, MAE=MAE))
    
    
    gnn_lg.train()
    log.info('Training the GNN with the line graph')
    logger.add_res('Training the GNN with the line graph')
    for epoch in range (args.max_epoch):
        
        t0 = time.time()
        
        if epoch != 0 and epoch % args.epoch_step == 0 :
            args.lr = args.lr * args.lrdamping
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
        loss, error = train_with_mnb(gnn_lg, train_set, args.task, criterion,
                                  optimizer_lg, args.cuda, args.batch_size, mean, std)
        
        dur = int(time.time() - t0)

        logger.add_train_loss(epoch+1,loss)
        logger.add_train_error(epoch+1,error)
        logger.add_epoch_time(epoch+1,dur)

        log.info('Epoch {} : Avg Error Ratio {:.3f}; Average Loss {:.3f} Time : {}'
          .format(epoch+1, error, loss, dur))
    
    avg_train_loss_lg = np.mean(np.array(logger.loss_train[args.max_epoch:]))
    avg_train_error_lg = np.mean(np.array(logger.error_train[args.max_epoch:]))
    training_time_lg = sum(logger.time_epoch[args.max_epoch:])
    MAE_lg = avg_train_error_lg * accuracy
    
    logger.add_res('Average training loss : {:.3f}'.format(avg_train_loss_lg))
    logger.add_res('Average training error : {:.3f}'.format(avg_train_error_lg))
    logger.add_res('Mean Absolute error : {:.3f}'.format(MAE_lg))
    logger.add_res('Training time : {} seconds'.format(training_time_lg))
    
    log.info('Avg Error Ratio {err:.3f}; Mean Average Error {MAE:.3f}'
          .format(err=avg_train_error_lg, MAE=MAE_lg))
    
    
    # Testing
    
    loss, error_ratio = test_mnb.test_with_mnb(gnn, valid_set, args.task, args.cuda, args.batch_size, mean, std, logger)
    loss_lg, error_ratio_lg = test_mnb.test_with_mnb(gnn_lg, valid_set, args.task, args.cuda, args.batch_size, mean, std, logger)
    
    logger.write_test_perf()
    
    logger.plot_train_logs()
    logger.plot_test_logs()
    
    MAE_test = error_ratio * accuracy
    MAE_test_lg = error_ratio_lg * accuracy
    
    log.info('Error ratio of the simple gnn : {:.5f} of the gnn using line graph : {:.5f} '
          .format(error_ratio, error_ratio_lg))
    
    return MAE_test, MAE_test_lg

  




