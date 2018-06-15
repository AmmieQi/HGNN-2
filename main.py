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


# Argument parser
parser = argparse.ArgumentParser(description='GNN on QM9 dataset')

add_arg = parser.add_argument  

add_arg('--datasetPath', dest = 'datasetPath', default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/debug.pickle')
add_arg('--logPath', dest = 'logPath', default='./log/gnn/')

# Optimization options
add_arg('--batchsize', dest='batchsize', help='mini-batch size',default=1)
add_arg('--maxepoch', dest='max_epoch', help='num epochs', default=300)
add_arg('--epochstep', dest='epoch_step', default=5)

add_arg('--lr', dest='lr', help='learning rate', type=float, default=0.002)
add_arg('--lrdamping', dest='lrdamping', help='learning rate damping', type=float, default=0.9)
add_arg('--momentum', dest='momentum', default=0.9)
add_arg('--lrdecay', dest='lrDecay', help='learning rate decay', default=0.6)
add_arg('--L', dest='L', help='epoch size', default=10)

# Model options
add_arg('--cuda',dest='cuda', help='Enables CUDA', default=True)
add_arg('--layers',dest='layers', help='input layers', default=8)
add_arg('--nfeatures', dest='nfeatures', help='feature maps', default=10)
add_arg('--J', dest='J', default=1)
add_arg('--dual', dest='dual', default=True)
add_arg('--task', dest='task', default=0)
add_arg('--preload', help='preload model', default='none')

    
def main():
    
    chem_acc = torch.tensor([0.1, 0.05, 0.043, 0.043, 0.043, 0.043,
                         0.043, 0.1, 10.0, 1.2, 0.043, 0.043, 0.0012])
    
    global args
    args = parser.parse_args()
    
    # Check if CUDA is enabled
    if args.cuda== True and torch.cuda.is_available():
        logging.warning('Working on GPU')
        dtype = torch.cuda.FloatTensor
        torch.cuda.manual_seed(0)
        chem_acc = chem_acc.cuda()
        
    else:
        logging.warning('Working on CPU')
        args.cuda = False
        dtype = torch.FloatTensor
        torch.manual_seed(0)

    # load and prepare data
    with open(args.datasetPath,'rb') as file :
        molecules = pickle.load(file)
    
    n = len(molecules)
    target = torch.zeros(n)
    for i in range (n):
        target[i] = molecules[i][2][args.task]
    mean = torch.mean(target)
    std = torch.std(target)
    target = normalize_data(target,mean,std)
    
    dim_input = molecules[0][0].size()[1]

    # model loading
    # create new model
    logging.warning('Creating new network')  
    
    if args.dual==False :
        gnn = GNN_simple(args.task, args.nfeatures, args.layers, dim_input,args.J)
        
    else:
        gnn = GNN_lg(args.task, args.nfeatures, args.layers, dim_input, args.J)
    logging.warning('New network created')
    
    # Optimizer
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    
    criterion = nn.MSELoss()
    
    if args.cuda == True :    
        gnn = gnn.cuda()
        criterion = criterion.cuda()
        
    # logger
    #logger = Logger(args.logPath)
    
    nloops = int(len(molecules) / args.batchsize)
    
    epoch_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()
    
    gnn.train()
    
    for epoch in range (args.max_epoch):
        
        t0 = time.time()
        
        if epoch != 0 and epoch % args.epoch_step == 0 :
            args.lr = args.lr * args.lrdamping
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
                

                
        for l in range (nloops):
            
            X, A, task = molecules[l]
            
            target = task[args.task]
            
            optimizer.zero_grad()
            
            if args.dual==False :
                W = graph_operators([X,A],args.J,False)
                
                if args.cuda == True:
                    X, W, target =  X.cuda(), W.cuda(), target.cuda()
                    gnn = gnn.cuda()
                
                # Resizing for the GNN : X must be (bs, n_feat, N)
                #                        W must be (bs, N, N, J+3)
                X = X.transpose(1,0)
                X = X.view(1,X.size()[0],X.size()[1])
                W = W.view(1,W.size()[0],W.size()[1],W.size()[2])
                output = gnn([X, W])
                
            else :
                W, WL, Pm, Pd = graph_operators([X,A],args.J,True)
                XL = torch.diag(WL[:,:,1])
                
                if args.cuda == True:
                    X, XL, W, WL, Pm, Pd, target =  X.cuda(), XL.cuda(), W.cuda(), WL.cuda(), Pm.cuda(), Pd.cuda(), target.cuda()
                    gnn = gnn.cuda()
                
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
                output = gnn([X, XL, W, WL, Pm, Pd])
            
            train_loss = criterion(output, target)
            #if (epoch == 0):
                #print("Molecule {} output {} target {}".format(l,output,target))

            # Logs
            losses.update(train_loss.item(), args.batchsize)
            error_ratio.update(evaluation(output, target).item(), args.batchsize)
            
            # compute gradient and do SGD step
            train_loss.backward()
            optimizer.step()
        
        end = time.time()
        epoch_time.update(int(end - t0))
        
        print('epoch {} loss : {:5f} error ratio : {:5f} time {}'.format(epoch+1, losses.avg, error_ratio.avg,epoch_time.val))

    MAE = error_ratio.avg * chem_acc[args.task]
    print('Final Avg Error Ratio {err.avg:.5f}; Mean Average Error {MAE:.5f}'
          .format(err=error_ratio, MAE=MAE))
    return MAE

  




