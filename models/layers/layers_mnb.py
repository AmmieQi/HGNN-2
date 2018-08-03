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

from functions.utils import graph_op, Pmul
from models.layers.batch_normalization import BN
from models.layers.gru_update import GRUUpdate, Identity

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
        
class layer_simple(nn.Module): 
    
    """ Layer (iteration) of the simple GNN with Relu non linearity """
    
    def __init__(self, feature_maps, J, gru):
        super(layer_simple, self).__init__()
        
        self.n_inputs = feature_maps[0]
        self.n_outputs = feature_maps[1]
        
        self.gop = graph_oper()
        self.cv1 = torch.nn.Conv1d(J*self.n_inputs, self.n_outputs, 1)
        self.cv2 = torch.nn.Conv1d(J*self.n_inputs, self.n_outputs, 1)
        if gru==True:
            self.update=GRUUpdate(self.n_inputs, 2*self.n_outputs)
        else:
            self.update=Identity()
        self.bn1 = BN(2*self.n_outputs)
            
        self._init_weights()

    def _init_weights(self, scale=0.1):
        layers = [self.cv1, self.cv2]
        for l in layers:
            l.weight.data.normal_(0, scale)
            l.bias.data.normal_(0, scale)

    def forward(self, state, N_batch, mask):
        """input :
            tensor X of size (bs, n_features, N)
	        tensor W of size (bs, N, N, J)"""
        
        X, W = state
        
        x1 = self.gop(W,X)
        y1 = self.cv1(x1)
        z1 = F.relu(y1)
        
        yl1 = self.cv2(x1)
        yl1 = F.relu(yl1)
        zb1 = torch.cat((yl1,z1),1)
        #zbn1 = zb1
        #zbu1 = self.update(x1,zb1)
        zbn1 = self.bn1(zb1, N_batch, mask)
        return (zbn1, W)
    

class layer_last(nn.Module):
    
    """ Final iteration (readout) giving the prediction for the single task """ 
    
    def __init__(self, feature_maps, J):
        super(layer_last, self).__init__()
        self.n_inputs = feature_maps[0]
        self.n_outputs = feature_maps[1]
        self.gop = graph_oper()
        self.fc = torch.nn.Conv1d(J*self.n_inputs, self.n_outputs, 1)
        self._init_weights()

    def _init_weights(self, scale=0.1):
        self.fc.weight.data.normal_(0, scale)
        self.fc.bias.data.normal_(0, scale)

    def forward(self, state, N_batch, mask):
        X, W = state
        x1 = self.gop(W,X)
        y1 = self.fc(x1)
        y = torch.sum(y1,dim=2)
        y = y.view(y.shape[0],self.n_outputs)
        #print(y.shape)
        return y
    

"""
class layer_with_lg(nn.Module): 
    
    Layer (iteration) of the GNN using the line graph with Relu non linearity 
        J is the total number of graph operators (including I and D)
    
    def __init__(self, feature_maps, J):
        super(layer_with_lg, self).__init__()
        
        self.n_inputs = feature_maps[0]
        self.n_edges = feature_maps[1] # number of features of the edge hidden state
        self.n_outputs = feature_maps[2]
        
        self.cv1 = torch.nn.Conv1d(J*self.n_inputs + 2*self.n_edges, self.n_outputs, 1)
        self.cv2 = torch.nn.Conv1d(J*self.n_inputs + 2*self.n_edges, self.n_outputs, 1)
        self.bn1 = BN(2*self.n_outputs)
        
        self.cv3 = torch.nn.Conv1d(J*self.n_edges  + 2*self.n_inputs, self.n_outputs, 1)
        self.cv4 = torch.nn.Conv1d(J*self.n_edges  + 2*self.n_inputs, self.n_outputs, 1)
        self.bn2 = BN(2*self.n_outputs)


    def forward(self, state, N_batch, mask, E_batch, mask_lg):
        
        input :
            tensor X of size (bs, n_features, N)
            tensor XL of size (bs, n_edges, M)
	        tensor W of size (bs, N, N, J)
            tensor WL of size (bs, M, M, J)
            tensor Pm of size (N,M)
            tensor Pd of size (N,M)

        X, XL, W, WL, Pm, Pd = state
        xa1 = graph_op(W,X)
        xb1 = Pmul(Pm,XL)
        xc1 = Pmul(Pd,XL)
        
        x1 = torch.cat((xa1,xb1,xc1),1)
        y1 = self.cv1(x1)
        yl1 = self.cv2(x1)
        #yl1 = F.relu(yl1)
        z1 = F.relu(y1)
        zb1 = torch.cat((yl1,z1),1)
        zbn1 = self.bn1(zb1, N_batch, mask)
        
        xda1 = graph_op(WL,XL)        
        xdb1 = Pmul(Pm.transpose(2,1),X)
        xdc1 = Pmul(Pd.transpose(2,1),X)
        xd1 = torch.cat((xda1,xdb1,xdc1),1)
        yd1 = self.cv3(xd1)
        zd1 = F.relu(yd1)
        ydl1 = self.cv4(xd1)
        #ydl1 = F.relu(ydl1)
        zdb1 = torch.cat((ydl1,zd1),1)
        zdbn1 = self.bn1(zdb1, E_batch, mask_lg)
        
        return (zbn1, zdbn1, W, WL, Pm, Pd)
"""

class layer_with_lg_1(nn.Module): 
    
    """ Layer (iteration) of the GNN using the line graph with Relu non linearity 
        J is the total number of graph operators (including I and D)
        update type 1 """
    
    def __init__(self, feature_maps, J):
        super(layer_with_lg_1, self).__init__()
        
        self.n_inputs = feature_maps[0]
        self.n_edges = feature_maps[1] # number of features of the edge hidden state
        self.n_outputs = feature_maps[2]
        
        self.gop = graph_oper()
        self.pmul = P_multi()
        self.cv1 = torch.nn.Conv1d(J*self.n_inputs + 2*self.n_edges, self.n_outputs, 1)
        self.cv2 = torch.nn.Conv1d(J*self.n_inputs + 2*self.n_edges, self.n_outputs, 1)
        self.bn1 = BN(2*self.n_outputs)
        
        self.cv3 = torch.nn.Conv1d(J*self.n_edges  + 4*self.n_outputs, self.n_outputs, 1)
        self.cv4 = torch.nn.Conv1d(J*self.n_edges  + 4*self.n_outputs, self.n_outputs, 1)
        self.bn2 = BN(2*self.n_outputs)
        
        self._init_weights()

    def _init_weights(self, scale=0.1):
        layers = [self.cv1, self.cv2, self.cv3, self.cv4]
        for l in layers:
            l.weight.data.normal_(0, scale)
            l.bias.data.normal_(0, scale)


    def forward(self, state, N_batch, mask, E_batch, mask_lg):
        
        """input :
            tensor X of size (bs, n_features, N)
            tensor XL of size (bs, n_edges, M)
	        tensor W of size (bs, N, N, J)
            tensor WL of size (bs, M, M, J)
            tensor Pm of size (N,M)
            tensor Pd of size (N,M)"""

        X, XL, W, WL, Pm, Pd = state
        xa1 = self.gop(W,X)
        xda1 = self.gop(WL,XL)
        
        xb1 = self.pmul(Pm,XL)
        xc1 = self.pmul(Pd,XL)
       
        x1 = torch.cat((xa1,xb1,xc1),1)
        y1 = self.cv1(x1)
        yl1 = self.cv2(x1)
        z1 = F.relu(y1)
        zb1 = torch.cat((yl1,z1),1)
        #zbn1 = zb1
        zbn1 = self.bn1(zb1, N_batch, mask)
    
        xdb1 = self.pmul(Pm.transpose(2,1),zbn1)
        xdc1 = self.pmul(Pd.transpose(2,1),zbn1)
            
        xd1 = torch.cat((xda1,xdb1,xdc1),1)
        yd1 = self.cv3(xd1)
        zd1 = F.relu(yd1)
        ydl1 = self.cv4(xd1)
        zdb1 = torch.cat((ydl1,zd1),1)
        #zdbn1 = zdb1
        zdbn1 = self.bn2(zdb1, E_batch, mask_lg)
        
        return (zbn1, zdbn1, W, WL, Pm, Pd)
        

class layer_with_lg_2(nn.Module): 
    
    def __init__(self, feature_maps, J):
        super(layer_with_lg_2, self).__init__()
        
        self.n_inputs = feature_maps[0]
        self.n_edges = feature_maps[1] # number of features of the edge hidden state
        self.n_outputs = feature_maps[2]
        
        self.gop = graph_oper()
        self.pmul = P_multi()
        self.cv1 = torch.nn.Conv1d(J*self.n_inputs + 4*self.n_outputs, self.n_outputs, 1)
        self.cv2 = torch.nn.Conv1d(J*self.n_inputs + 4*self.n_outputs, self.n_outputs, 1)
        self.bn1 = BN(2*self.n_outputs)
        
        self.cv3 = torch.nn.Conv1d(J*self.n_edges  + 2*self.n_inputs, self.n_outputs, 1)
        self.cv4 = torch.nn.Conv1d(J*self.n_edges  + 2*self.n_inputs, self.n_outputs, 1)
        self.bn2 = BN(2*self.n_outputs)
        
        self._init_weights()

    def _init_weights(self, scale=0.1):
        layers = [self.cv1, self.cv2, self.cv3, self.cv4]
        for l in layers:
            l.weight.data.normal_(0, scale)
            l.bias.data.normal_(0, scale)


    def forward(self, state, N_batch, mask, E_batch, mask_lg):
        
        """input :
            tensor X of size (bs, n_features, N)
            tensor XL of size (bs, n_edges, M)
	        tensor W of size (bs, N, N, J)
            tensor WL of size (bs, M, M, J)
            tensor Pm of size (N,M)
            tensor Pd of size (N,M)"""
        
        X, XL, W, WL, Pm, Pd = state
        xa1 = self.gop(W,X)
        xda1 = self.gop(WL,XL)
        
        xdb1 = self.pmul(Pm.transpose(2,1),X)
        xdc1 = self.pmul(Pd.transpose(2,1),X)
        xd1 = torch.cat((xda1,xdb1,xdc1),1)
        yd1 = self.cv3(xd1)
        zd1 = F.relu(yd1)
        ydl1 = self.cv4(xd1)
        #ydl1 = F.relu(ydl1)
        zdb1 = torch.cat((ydl1,zd1),1)
        zdbn1 = self.bn2(zdb1, E_batch, mask_lg)
        
        xb1 = self.pmul(Pm,zdbn1)
        xc1 = self.pmul(Pd,zdbn1)  
        x1 = torch.cat((xa1,xb1,xc1),1)
        y1 = self.cv1(x1)
        yl1 = self.cv2(x1)
        #yl1 = F.relu(yl1)
        z1 = F.relu(y1)
        zb1 = torch.cat((yl1,z1),1)
        zbn1 = self.bn1(zb1, N_batch, mask)
        
        return (zbn1, zdbn1, W, WL, Pm, Pd)
    
   
class layer_with_lg_3(nn.Module): 
    
    
    def __init__(self, feature_maps, J):
        super(layer_with_lg_3, self).__init__()
        
        self.n_inputs = feature_maps[0]
        self.n_edges = feature_maps[1] # number of features of the edge hidden state
        self.n_outputs = feature_maps[2]
        
        self.gop = graph_oper()
        self.pmul = P_multi()
        self.cv1 = torch.nn.Conv1d(J*self.n_inputs + 2*self.n_edges, self.n_outputs, 1)
        self.cv2 = torch.nn.Conv1d(J*self.n_inputs + 2*self.n_edges, self.n_outputs, 1)
        self.bn1 = BN(2*self.n_outputs)
        
        self.cv3 = torch.nn.Conv1d(J*self.n_edges  + 2*self.n_inputs, self.n_outputs, 1)
        self.cv4 = torch.nn.Conv1d(J*self.n_edges  + 2*self.n_inputs, self.n_outputs, 1)
        self.bn2 = BN(2*self.n_outputs)
        
        self._init_weights()

    def _init_weights(self, scale=0.1):
        layers = [self.cv1, self.cv2, self.cv3, self.cv4]
        for l in layers:
            l.weight.data.normal_(0, scale)
            l.bias.data.normal_(0, scale)


    def forward(self, state, N_batch, mask, E_batch, mask_lg):
        
        """input :
            tensor X of size (bs, n_features, N)
            tensor XL of size (bs, n_edges, M)
	        tensor W of size (bs, N, N, J)
            tensor WL of size (bs, M, M, J)
            tensor Pm of size (N,M)
            tensor Pd of size (N,M)"""

        X, XL, W, WL, Pm, Pd = state
        xa1 = self.gop(W,X)
        xda1 = self.gop(WL,XL)
        
        xb1 = self.pmul(Pm,XL)
        xc1 = self.pmul(Pd,XL)
        
        xdb1 = self.pmul(Pm.transpose(2,1),X)
        xdc1 = self.pmul(Pd.transpose(2,1),X)
                  
        x1 = torch.cat((xa1,xb1,xc1),1)
        y1 = self.cv1(x1)
        yl1 = self.cv2(x1)
        #yl1 = F.relu(yl1)
        z1 = F.relu(y1)
        zb1 = torch.cat((yl1,z1),1)
        zbn1 = self.bn1(zb1, N_batch, mask)
        
        xd1 = torch.cat((xda1,xdb1,xdc1),1)
        yd1 = self.cv3(xd1)
        zd1 = F.relu(yd1)
        ydl1 = self.cv4(xd1)
        #ydl1 = F.relu(ydl1)
        zdb1 = torch.cat((ydl1,zd1),1)
        zdbn1 = self.bn2(zdb1, E_batch, mask_lg)
        
        return (zbn1, zdbn1, W, WL, Pm, Pd)
    
    
class layer_last_lg(nn.Module): 
    
    """ Last iteration of the GNN using the line graph  """
    
    def __init__(self, feature_maps, J):
        super(layer_last_lg, self).__init__()
        self.n_inputs = feature_maps[0]
        self.n_outputs = feature_maps[1]
        self.gop = graph_oper()
        self.pmul = P_multi()
        self.fc = torch.nn.Conv1d((J+2)*self.n_inputs, self.n_outputs, 1)
        
        self._init_weights()

    def _init_weights(self, scale=0.1):
        self.fc.weight.data.normal_(0, scale)
        self.fc.bias.data.normal_(0, scale)

    def forward(self, state, N_batch, mask):
        X, XL, W, WL, Pm, Pd = state
        xa1 = self.gop(W,X)
        xb1 = self.pmul(Pm,XL)
        xc1 = self.pmul(Pd,XL)
        x1 = torch.cat((xa1,xb1,xc1),1)
        y1 = self.fc(x1)
        y = torch.sum(y1,dim=2)
        y = y.view(y.shape[0],self.n_outputs)
        return y
    
    
class graph_oper(nn.Module):
    def __init__(self):
        super(graph_oper, self).__init__()
        
    def forward(self, A, X):
        bs = A.size()[0]
        N = A.size()[1]
        J = A.size()[3]
        n_feat = X.size()[1]
        
        output = torch.zeros(bs, J*n_feat, N).type(dtype) 
        
        for b in range (bs) :
            for j in range (J) :               
                Aslice =  A[b,:,:,j] # N x N
                Xslice = X[b,:,:].transpose(1,0) # N x n_feat
                AX = torch.mm(Aslice, Xslice) # dimension N x n_feat
                
                output[b,n_feat*j:n_feat*(j+1),:] = AX.transpose(1,0)
     
        return output
    
    
class P_multi(nn.Module):
    def __init__(self):
        super(P_multi, self).__init__()
        
    def forward(self, P, X):
        bs = P.size()[0]
        N = P.size()[1]
        #M = P.size()[2]
        n_feat = X.size()[1]
        
        output = torch.zeros(bs, n_feat, N).type(dtype) 
        
        for b in range (bs) :
            
            Pslice =  P[b,:,:] # N x M
            Xslice = X[b,:,:].transpose(1,0) # M x n_feat
            AX = torch.mm(Pslice, Xslice) # dimension N x n_feat
            
            output[b,:,:] = AX.transpose(1,0)
     
        return output
        