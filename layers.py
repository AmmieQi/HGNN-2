#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:50:24 2018

@author: sulem
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from utils import *


        
class layer_simple(nn.Module): 
    
    """ Layer (iteration) of the simple GNN with Relu non linearity """
    
    def __init__(self, feature_maps, J):
        super(layer_simple, self).__init__()
        
        self.n_inputs = feature_maps[0]
        self.n_outputs = feature_maps[1]

        self.cv1 = torch.nn.Conv1d(J*self.n_inputs, self.n_outputs, 1)
        self.cv2 = torch.nn.Conv1d(J*self.n_inputs, self.n_outputs, 1)
        #self.bn1 = nn.BatchNorm1d(2*self.n_outputs)

    def forward(self, input):
        """input :
            tensor X of size (bs, n_features, N)
	        tensor W of size (bs, N, N, J)"""
        
        X, W = input
        
        x1 = graph_op(W,X)
        y1 = self.cv1(x1)
        z1 = F.relu(y1)
        
        yl1 = self.cv2(x1)
        zb1 = torch.cat((yl1,z1),1)
        #zc1 = self.bn1(zb1)
        
        return (zb1, W)
    

class layer_last(nn.Module):
    
    """ Final iteration (readout) giving the prediction for the single task """ 
    
    def __init__(self, feature_maps, J):
        super(layer_last, self).__init__()
        self.n_inputs = feature_maps[0]
        self.fc = torch.nn.Conv1d(J*self.n_inputs, 1, 1)

    def forward(self, input):
        X, W = input
        x1 = graph_op(W,X)
        y1 = self.fc(x1)
        y = torch.sum(y1.squeeze())
        return y
    
    
class layer_with_lg(nn.Module): 
    
    """ Layer (iteration) of the GNN using the line graph with Relu non linearity 
        J is the total number of graph operators (including I and D) """
    
    def __init__(self, feature_maps, J):
        super(layer_with_lg, self).__init__()
        
        self.n_inputs = feature_maps[0]
        self.n_edges = feature_maps[1] # number of features of the edge hidden state
        self.n_outputs = feature_maps[2]
        
        self.cv1 = torch.nn.Conv1d(J*self.n_inputs + 2*self.n_edges, self.n_outputs, 1)
        self.cv2 = torch.nn.Conv1d(J*self.n_inputs + 2*self.n_edges, self.n_outputs, 1)
        #self.bn1 = nn.BatchNorm1d(2*self.n_oX = X.view(1,X.size()[0],X.size()[1])utputs)
        
        self.cv3 = torch.nn.Conv1d(J*self.n_edges  + 2*self.n_inputs, self.n_outputs, 1)
        self.cv4 = torch.nn.Conv1d(J*self.n_edges  + 2*self.n_inputs, self.n_outputs, 1)
        #self.bn2 = nn.BatchNorm1d(2*self.n_outputs)


    def forward(self, input):
        
        """input :
            tensor X of size (bs, n_features, N)
            tensor XL of size (bs, n_edges, M)
	        tensor W of size (bs, N, N, J)
            tensor WL of size (bs, M, M, J)
            tensor Pm of size (N,M)
            tensor Pd of size (N,M)"""

        X, XL, W, WL, Pm, Pd = input
        xa1 = graph_op(W,X)
        xb1 = Pmul(Pm,XL)
        xc1 = Pmul(Pd,XL)
        
        x1 = torch.cat((xa1,xb1,xc1),1)
        y1 = self.cv1(x1)
        yl1 = self.cv2(x1)
        z1 = F.relu(y1)
        zb1 = torch.cat((yl1,z1),1)
        #zc1 = self.bn1(zb1)
        
        xda1 = graph_op(WL,XL)        
        xdb1 = Pmul(Pm.transpose(2,1),X)
        xdc1 = Pmul(Pd.transpose(2,1),X)
        xd1 = torch.cat((xda1,xdb1,xdc1),1)
        yd1 = self.cv3(xd1)
        zd1 = F.relu(yd1)
        ydl1 = self.cv4(xd1)
        zdb1 = torch.cat((ydl1,zd1),1)
        #zdc1 = self.bn2(zdb1)
        
        return (zb1, zdb1, W, WL, Pm, Pd)
    
    
class layer_last_lg(nn.Module): 
    
    """ Last iteration of the GNN using the line graph  """
    
    def __init__(self, feature_maps, J):
        super(layer_last_lg, self).__init__()
        self.n_inputs = feature_maps[0]
        self.fc = torch.nn.Conv1d((J+2)*self.n_inputs, 1, 1)

    def forward(self, input):
        X, XL, W, WL, Pm, Pd = input
        xa1 = graph_op(W,X)
        xb1 = Pmul(Pm,XL)
        xc1 = Pmul(Pd,XL)
        x1 = torch.cat((xa1,xb1,xc1),1)
        y1 = self.fc(x1)
        y = torch.sum(y1.squeeze())
        
        return y
    