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
from train import *
from test import *

# Argument parser
parser = argparse.ArgumentParser(description='GNN on QM9 dataset')

add_arg = parser.add_argument  

add_arg('--train_path', dest = 'train_path', default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/train_set_5000.pickle')
add_arg('--valid_path', dest = 'valid_path', default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/valid_set_500.pickle')
add_arg('--logPath', dest = 'log_path', default='/misc/vlgscratch4/BrunaGroup/sulem/chem/log/testing/')

# Optimization options
#add_arg('--batchsize', dest='batchsize', help='mini-batch size',default=1)
add_arg('--maxepoch', dest='max_epoch', help='num epochs', default=1)
add_arg('--epochstep', dest='epoch_step', default=3)

add_arg('--lr', dest='lr', help='learning rate', type=float, default=0.001)
add_arg('--lrdamping', dest='lrdamping', help='learning rate damping', type=float, default=0.9)
add_arg('--momentum', dest='momentum', default=0.9)
add_arg('--lrdecay', dest='lrDecay', help='learning rate decay', default=0.6)
#add_arg('--L', dest='L', help='epoch size', default=10)

# Model options
add_arg('--cuda',dest='cuda', help='Enables CUDA', default=True)
add_arg('--layers',dest='layers', help='input layers', default=5)
add_arg('--nfeatures', dest='nfeatures', help='feature maps', default=10)
add_arg('--J', dest='J', default=1)
add_arg('--task', dest='task', default=0)

    
def test_hyperparameters():
    
    chem_acc = torch.tensor([0.1, 0.05, 0.043, 0.043, 0.043, 0.043,
                         0.043, 0.1, 10.0, 1.2, 0.043, 0.043, 0.0012])
    
    global args
    args = parser.parse_args()
    
    accuracy = chem_acc[args.task]
    
    # logger
    logger = Logger(args.log_path)
    # Write experiment settings
    logger.write_settings(args)
    logger.add_info('Chemical accuracy for task {} : {}'.format(args.task, accuracy))
    
    # Check if CUDA is enabled
    if args.cuda== True and torch.cuda.is_available():
        logging.warning('Working on GPU')
        dtype = torch.cuda.FloatTensor
        torch.cuda.manual_seed(0)
        
    else:
        logging.warning('Working on CPU')
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
    t_stats = target_stats(train_target)
    train_target = normalize_data(train_target,t_stats[2],t_stats[3])
    
    for i in range (Nvalid):
        valid_target[i] = valid_set[i][2][args.task]
    v_stats = target_stats(valid_target)
    valid_target = normalize_data(valid_target,v_stats[2],v_stats[3])
    
    logger.add_info('Stats on the training set task [min, max, mean, std] : ' + str(t_stats))
    logger.add_info('Stats on the validation set task [min, max, mean, std] : ' + str(v_stats))
    
    dim_input = train_set[0][0].size()[1]

    # Creates 2 GNNs : 1 not using the line graph, 1 using it       
    gnn = GNN_simple(args.task, args.nfeatures, args.layers, dim_input,args.J)
    logger.add_model('gnn simple')
    gnn_lg = GNN_lg(args.task, args.nfeatures, args.layers, dim_input, args.J)
    logger.add_model('gnn with LG')
    logging.warning('2 networks created')
    
    # Optimizers
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    optimizer_lg = torch.optim.Adamax(gnn_lg.parameters(), lr=args.lr)
    
    if args.cuda == True :    
        gnn = gnn.cuda()
        gnn_lg = gnn_lg.cuda()
    
    # Training the 2 models successively
    
    gnn.train()
    
    logger.add_res('Training the GNN without line graph')
    for epoch in range (args.max_epoch):
        
        if epoch != 0 and epoch % args.epoch_step == 0 :
            args.lr = args.lr * args.lrdamping
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
        loss, error, dur = train_model(gnn, train_set, args.task, False, args.J, optimizer_lg, args.cuda)

        logger.add_train_loss(epoch+1,loss)
        logger.add_train_error(epoch+1,error)
        logger.add_epoch_time(epoch+1,dur)

        print('Epoch {} : Avg Error Ratio {:.3f}; Average Loss {:.3f} Time : {}'
          .format(epoch+1, error, loss, dur))
    
    avg_train_loss = np.mean(np.array(logger.loss_train))
    avg_train_error = np.mean(np.array(logger.error_train))
    training_time = sum(logger.time_epoch)
    MAE = avg_train_error * accuracy
    
    logger.add_res('Average training loss : {:.3f}'.format(avg_train_loss))
    logger.add_res('Average training error : {:.3f}'.format(avg_train_error))
    logger.add_res('Mean Absolute error : {:.3f}'.format(MAE))
    logger.add_res('Training time : {} seconds'.format(training_time))
    
    print('Avg Error Ratio {err:.3f}; Mean Average Error {MAE:.3f}'
          .format(err=avg_train_error, MAE=MAE))
    
    
    gnn_lg.train()
    
    logger.add_res('Training the GNN with the line graph')
    for epoch in range (args.max_epoch):
        
        if epoch != 0 and epoch % args.epoch_step == 0 :
            args.lr = args.lr * args.lrdamping
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
        loss, error, time = train_model(gnn_lg, train_set, args.task, True, args.J, optimizer, args.cuda)

        logger.add_train_loss(epoch+1,loss)
        logger.add_train_error(epoch+1,error)
        logger.add_epoch_time(epoch+1,time)

        print('Epoch {} : Avg Error Ratio {:.3f}; Average Loss {:.3f} Time : {}'
          .format(epoch+1, error, loss, time))
    
    print(logger.loss_train[args.max_epoch:])
    print(logger.error_train[args.max_epoch:])
    
    avg_train_loss_lg = np.mean(np.array(logger.loss_train[args.max_epoch:]))
    avg_train_error_lg = np.mean(np.array(logger.error_train[args.max_epoch:]))
    training_time_lg = sum(logger.time_epoch[Ntrain:])
    MAE_lg = avg_train_error_lg * accuracy
    
    logger.add_res('Average training loss : {:.3f}'.format(avg_train_loss_lg))
    logger.add_res('Average training error : {:.3f}'.format(avg_train_error_lg))
    logger.add_res('Mean Absolute error : {:.3f}'.format(MAE_lg))
    logger.add_res('Training time : {} seconds'.format(training_time_lg))
    
    print('Avg Error Ratio {err:.3f}; Mean Average Error {MAE:.3f}'
          .format(err=avg_train_error_lg, MAE=MAE_lg))
    
    
    # Testing
    
    error_ratio, test_time = test_model(gnn, valid_set, args.task, False, args.J, args.cuda,logger)
    error_ratio_lg, test_time_lg = test_model(gnn_lg, valid_set, args.task, True, args.J, args.cuda,logger)
    
    logger.add_test_time(test_time)
    logger.add_test_time(test_time_lg)
    logger.write_test_perf()
    
    logger.plot_train_logs()
    logger.plot_test_logs()
    
    MAE_test = error_ratio * accuracy
    MAE_test_lg = error_ratio_lg * accuracy
    
    print('Mean Average Error of the simple gnn : {:.5f} of the gnn using line graph : {:.5f} '
          .format(MAE_test, MAE_test_lg))
    
    return MAE_test, MAE_test_lg

  




