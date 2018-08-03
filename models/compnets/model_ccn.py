#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:07:12 2018

@author: sulem
"""

import torch
#import functions.contract18 # for cuda contraction
import torch.nn.functional as Func
from torch.autograd import Variable
import torch.nn as nn
#from contraction import collapse6to3
from functions.utils_ccn import CompnetUtils


class CCN_1D(nn.Module):
    def __init__(self, input_feats, n_outputs=1, hidden_size=2, layers=2, cudaflag=False):
        super(CCN_1D, self).__init__()
        self.input_feats = input_feats
        self.n_outputs = n_outputs
        self.hidden_size =  hidden_size
        self.num_contractions = 2
        self.layers = layers

        self.utils = CompnetUtils(cudaflag)
        self.w1 = nn.Linear(input_feats * self.num_contractions, hidden_size)
        for i in range(layers-1):
            module = nn.Linear(hidden_size * self.num_contractions, hidden_size)
            self.add_module('w{}'.format(i + 2), module)
        self.fc = nn.Linear(self.layers * hidden_size + input_feats, self.n_outputs)
        self._init_weights()

    def _init_weights(self, scale=0.1):
        layers = [self._modules['w{}'.format(i+1)] for i in range(self.layers)] + [self.fc]
        for l in layers:
            l.weight.data.normal_(0, scale)
            l.bias.data.normal_(0, scale)

    def forward(self, X, adj):
        F = []
        cur = self.utils.get_F0_1D(X, adj)
        F.append(cur)
        #k = [F_0[i].shape[0] for i in range (len(F_0))]
        #d = [F_0[i].shape[1] for i in range (len(F_0))]
        #print("Dimensions of the activation tensor on layer 0 : {} , {}".format(k,d))
        
        for i in range(self.layers):
            #print(len(cur))
            cur = self.utils.update_F_1D(cur, self._modules['w{}'.format(i+1)])
            F.append(cur)
        
        #n = len(F_1)
        #dim = [F_1[i].shape for i in range (len(F_1))]
        #print("Dimensions of the activation tensor on layer 1 : nb of nodes {} , dimensions {}".format(n,dim))
        
        #n = len(F_2)
        #dim = [F_2[i].shape for i in range (len(F_2))]
        #print("Dimensions of the activation tensor on layer 2 : nb of nodes {} , dimensions {}".format(n,dim))
        summed = [sum([v.sum(0) for v in f]) for f in F]
        graph_feat = torch.cat(summed, 0)
        #print("Graph feature tensor : dimensions {} values {}".format(graph_feat.shape,graph_feat))
        return self.fc(graph_feat)
    
    

class CCN_2D(nn.Module):
    def __init__(self, input_feats=2, n_outputs=1, hidden_size=2, layers=2, cudaflag=True):
        super(CCN_2D, self).__init__()
        self.input_feats = input_feats
        self.n_outputs = n_outputs
        self.hidden_size = 2
        self.num_contractions = 18
        self.layers = layers
        self.cudaflag = cudaflag

        self.utils = CompnetUtils(cudaflag)
        self.w1 = nn.Linear(input_feats * self.num_contractions, hidden_size)
        for i in range(layers-1):
            module = nn.Linear(hidden_size * self.num_contractions, hidden_size)
            self.add_module('w{}'.format(i + 2), module)
        self.fc = nn.Linear(self.layers * hidden_size + input_feats, self.n_outputs)
        self._init_weights()
        
    def _init_weights(self, scale=0.1):
        layers = [self._modules['w{}'.format(i+1)] for i in range(self.layers)]
        for l in layers:
            l.weight.data.normal_(0, scale)
        self.fc.weight.data.normal_(0, scale)
        self.fc.weight.data.normal_(0, scale * 5)
        
    def forward(self, X, adj):
        F = []
        cur = self.utils.get_F0(X, adj)
        F.append(cur)
        
        for i in range(self.layers):
            cur = self.utils.update_F(cur, self._modules['w{}'.format(i+1)])
            F.append(cur)
        
        summed = [sum([v.sum(0).sum(0) for v in f]) for f in F]
        graph_feat = torch.cat(summed, 0)
        #print("Graph feature tensor : dimensions {} values {}".format(graph_feat.shape,graph_feat))
        return self.fc(graph_feat)