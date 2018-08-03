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
from functions import utils, logs
from preprocessing import loading
from scripts import train_mnb, test_mnb

# Logging settings
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

###############################################################################

# Argument parser
global parser
parser = argparse.ArgumentParser(description='GNN on QM9 dataset')

add_arg = parser.add_argument


"""
add_arg('--train_path', dest = 'train_path', 
        default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/train.pickle')
add_arg('--valid_path', dest = 'valid_path',
        default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/valid.pickle')
add_arg('--test_path', dest = 'test_path',
        default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/test.pickle')
"""

add_arg('--data_path', dest = 'data_path', 
        default='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_0.pickle')
add_arg('--log_path', dest = 'log_path', default=None)

add_arg('--train', dest='train', help='training',default=True)
add_arg('--val', dest='val', help='validate',default=True)
add_arg('--test', dest='test', help='testing',default=True)

# Optimization options
add_arg('--bs', dest='batch_size', help='mini-batch size',default=30,type=int)
add_arg('--epochs', dest='max_epoch', help='num epochs', default=40,type=int)
add_arg('--step', dest='epoch_step', default=5,type=int)

add_arg('--optim', dest='optim', help='Optimization algorithm', type=str, default='adamax')
add_arg('--lr', dest='lr', help='learning rate', type=float, default=0.0003)
add_arg('--lrdamping', dest='lrdamping', help='learning rate damping',
        type=float, default=0.90)
add_arg('--momentum', dest='momentum', default=0.9,type=float)
add_arg('--shuffle', dest='shuffle', default=False)

# Model options
add_arg('--model', dest = 'model_path',
        default=None)
add_arg('--lg',dest='lg', help='With LG', default=False)
add_arg('--update',dest='update', help='Order of updates', default=1,type=int)
add_arg('--cuda',dest='cuda', help='Enables CUDA', default=True,type=bool)
add_arg('--L',dest='layers', help='input layers', default=15,type=int)
add_arg('--h', dest='nfeatures', help='feature maps', default=1,type=int)
add_arg('--J', dest='J', default=1,type=int)
add_arg('--task', dest='task', default=0,type=int)
add_arg('--sp', dest='spatial', default=False,help='Spatial info in the input',type=bool)
add_arg('--pc', dest='charge', default=False,help='Partial charge in the input',type=bool)
add_arg('--ninput', dest='dim_input', default=5,type=int)

###############################################################################


    
def main():
    
    global args
    args = parser.parse_args()
    
    # Setting log path
    if args.log_path == None:
        log_path = ('log/qm9/lg_' + str(args.lg) + '_up_' + str(args.update) + '_bs_' 
                    + str(args.batch_size) + '_ep_' + str(args.max_epoch) + '_st_' + str(args.epoch_step)
                    + '_op_' + str(args.optim) + '_lr_' + str(args.lr) + '_da_' + str(args.lrdamping)
                    + '_L_' + str(args.layers) + '_h_' + str(args.nfeatures) + '_ta_' + str(args.task)
                    + '_' + str(time.time())[-3:] + '.pickle'
        )
        args.log_path = log_path
    log.info("Log path : " + log_path)
    
    # Initializing logger
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
        
    # Loading population statistics for the task
    stats_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/target_stat.pickle'
    with open(stats_path,'rb') as file :
        M, S, A = pickle.load(file)
    mean = M[args.task].item()
    std = S[args.task].item()
    accuracy = A[args.task].item()
    
    # Loading experiment sets
    
    if args.spatial==True and args.charge==False:
        args.dim_input=8
        args.data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_sp.pickle'
    elif args.spatial==True and args.charge==True:
        args.dim_input=9
        args.data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_sp_ch.pickle'
    elif args.spatial==False and args.charge==True:
        args.dim_input=6
        args.data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_ch.pickle'
    else:
        args.dim_input=5
        args.data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9.pickle'
        
    logging.info("Loading data...")
    with open(args.data_path,'rb') as file :
            data_set = pickle.load(file)
            
    train_set, valid_set, test_set = loading.prepare_experiment_sets(data_set,
                                                                     args.shuffle)
    if args.train==True:
        Ntrain = len(train_set)   
        log.info("Number of training instances : " + str(Ntrain))
        logger.add_info('Training set size : ' + str(Ntrain))
    
    if args.val==True:
        Nvalid = len(valid_set) 
        log.info("Number of validation instances : " + str(Nvalid))
        logger.add_info('Validation set size : ' + str(Nvalid))
            
    if args.test==True:
        Ntest = len(test_set) 
        log.info("Number of test instances : " + str(Ntest))
        logger.add_info('Test set size : ' + str(Ntest))

    # Creating or loading model
    if args.model_path != None:
        gnn = torch.load(args.model_path)
        log.info('Network loaded')
    else:
        if args.lg == False :     
            gnn = model_mnb.GNN_simple(args.task, args.nfeatures, args.layers,
                                       args.dim_input, 1, args.J)
            logger.add_model('gnn simple')
        else:
            gnn = model_mnb.GNN_lg(args.task, args.nfeatures, args.layers,
                                   args.dim_input, args.J, 1, args.update)
            logger.add_model('gnn with LG')
        log.info('Network created')
    
    # Criterion and optimizer
    criterion = nn.MSELoss()
        
    if args.cuda == True :    
        gnn = gnn.cuda()
        criterion = criterion.cuda()

    # Training
    
    if args.train==True:
        gnn.train()
        
        log.info('Training the GNN')
        logger.add_res('Training phase')
        
        run_loss = utils.RunningAverage()
        run_error = utils.RunningAverage()
        
        for epoch in range (args.max_epoch):
            
            t0 = time.time()
            
            optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
            
            loss, error = train_mnb.train_with_mnb(gnn, train_set, args.task, criterion,
                                         optimizer, args.cuda, args.batch_size, mean, std)
            
            v_loss, v_error = test_mnb.test_with_mnb(gnn, valid_set, args.task,
                                                     criterion, args.cuda, args.batch_size,
                                                     mean, std, logger)
            
            t_loss, t_error = test_mnb.test_with_mnb(gnn, test_set, args.task,
                                         criterion, args.cuda, args.batch_size,
                                         mean, std, logger)
            
            dur = int(time.time() - t0)
            
            run_loss.update(loss)
            run_error.update(error)
            
            if epoch != 0 and epoch % args.epoch_step == 0 :
                args.lr = args.lr * args.lrdamping
            
            logger.add_epoch_logs(epoch+1,run_loss.val, run_error.val, v_loss,
                                  v_error, t_loss, t_error, dur)
            log.info('Epoch {} : Train loss {:.3f} error {:.3f} Time : {}'
              .format(epoch+1, run_error.val, run_loss.val, dur))
            log.info('Validation loss {:.3f} error {:.3f}'
              .format(v_loss, v_error))
            log.info('Test loss {:.3f} error {:.3f}'
              .format(t_loss, t_error))
        
        training_time = sum(logger.time_epoch) // 60
        ratio = run_error.val / accuracy
        
        v_loss = logger.loss_valid[-1]
        v_error = logger.error_valid[-1]
        t_loss = logger.loss_test[-1]
        t_error = logger.error_test[-1]
        
        logger.add_train_info(run_loss.val, run_error.val, ratio, training_time)
        logger.add_valid_perf(v_loss, v_error, v_error/accuracy)
        logger.add_test_perf(t_loss, t_error, t_error/accuracy)
        log.info('Training finished : Duration {} minutes, Loss {:.3f}, MAE {:.3f}, Error ratio {:.3f}'
              .format(training_time, run_loss.val, run_error.val, ratio))
        log.info('Validation loss {:.3f} error {:.3f}'.format(v_loss, v_error))
        log.info('Test loss {:.3f} error {:.3f}'.format(t_loss, t_error))
        
        logger.plot_loss()
        logger.plot_error()
        
        logger.save_model(gnn)
        
        
        
    """
    # Validating
    
    if args.val==True:
        log.info('Evaluating on the validation set...')
        logger.add_res('Validation phase')
        val_loss, val_error = test_mnb.test_with_mnb(gnn, valid_set, args.task,
                                                     criterion, args.cuda, args.batch_size,
                                                     mean, std, logger)
        ratio_val = val_error / accuracy
        log.info('Validation finished : Avg loss {:.3f}, Mean Average Error {:.3f}, Error ratio {:.3f}'
                 .format(val_loss, val_error, ratio_val))
        logger.add_test_perf(val_loss, val_error, ratio_val)
        
        logger.plot_train_logs()
        logger.plot_test_logs()    
        
    # Testing
    if args.test==True:
        log.info('Evaluating on the test set...')
        logger.add_res('Test phase')
        test_loss, test_error = test_mnb.test_with_mnb(gnn, test_set, args.task, criterion,
                                                       args.cuda, args.batch_size,
                                                       mean, std, logger)
        ratio_test = test_error / accuracy
        log.info('Test finished : Avg loss {:.3f}, Mean Average Error {:.3f}, Error ratio {:.3f}'
                 .format(test_loss, test_error, ratio_test))
        logger.add_test_perf(test_loss, test_error, ratio_test)
        
        logger.plot_train_logs()
        #logger.plot_test_logs() 
        
        return test_error, ratio_test
    """

if __name__ == '__main__':
    main()
  




