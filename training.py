################Import Packages#######################
from __future__ import division, print_function
from typing import Dict, SupportsRound, Tuple, Any
from os import PathLike
from pathlib import Path
import matplotlib.pyplot as plt
import torch,gc
from torch.autograd import grad
from torch.autograd import Variable
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import SimpleITK as sitk
import os, glob
import json
import subprocess
import sys
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import TensorDataset, DataLoader

import random
import yaml
from tqdm import tqdm, trange

from easydict import EasyDict as edict
import json
import cv2
import pickle

import numpy as np
from numpy import zeros, newaxis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
    
CUDA_LAUNCH_BLOCKING=1

    
################Parameter Loading#######################
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None
    
para = read_yaml('./parameters.yml')

xDim = para.data.x 
yDim = para.data.y
zDim = para.data.z

def loss_Reg(y_pred):
        ### For 3D reg ###
        # dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        # dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        # dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        # dy = dy * dy
        # dx = dx * dx
        # dz = dz * dz
        # d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        # grad = d / 3.0

        ### For 2D reg ###
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        dy = dy * dy
        dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy) 
        grad = d / 2.0
        
        return grad
    
##################Data Loading##########################
readfilename = './2DShape/data' + '.json'
datapath = './2DShape/'
data = json.load(open(readfilename, 'r'))
outputs = []
keyword = 'train'
# outputs = np.array(outputs)

print ('*****Dataset Loading*****')
for i in trange (0,len(data[keyword])):
    filename_src = datapath + data[keyword][i]['source']
    itkimage_src = sitk.ReadImage(filename_src)
    source_scan = sitk.GetArrayFromImage(itkimage_src)
    source_scan = cv2.resize(source_scan, (128, 128))
    source_scan = (source_scan - np.min(source_scan)) / (np.max(source_scan) - np.min(source_scan))
    
    filename_tar = datapath + data[keyword][i]['target']
    itkimage_tar = sitk.ReadImage(filename_tar)
    target_scan = sitk.GetArrayFromImage(itkimage_tar)
    target_scan = cv2.resize(target_scan, (128, 128))
    target_scan = (target_scan - np.min(target_scan)) / (np.max(target_scan) - np.min(target_scan))
    
    pair = np.concatenate((source_scan[newaxis,:], target_scan[newaxis,:]), axis=0)
    outputs.append(pair)

train = torch.FloatTensor(outputs)
print ('Dataset Size: ', train.shape)



#################Network optimization########################
from network import DiffeoDense

net = []
for i in range(3):
    temp = DiffeoDense(inshape = (xDim,yDim),
				 nb_unet_features= [[16, 32],[ 32, 32, 16, 16]],
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res= True)
    net.append(temp)
net = net[0].to(dev)

class TDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index] if self.labels is not None else None
        return data, label
    
train_labels = json.load(open('./2DShape/data.json','r'))
train_set = TDataset(train, [d['label'] for d in train_labels['train']])
trainloader = torch.utils.data.DataLoader(train_set, batch_size = para.solver.batch_size, shuffle=True, num_workers=1)
sampleloader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle=True, num_workers=1)

running_loss = 0 
running_loss_val = 0
template_loss = 0
printfreq = 1
sigma = 0.02
repara_trick = 0.0
loss_array = torch.FloatTensor(para.solver.epochs,1).fill_(0)
loss_array_val = torch.FloatTensor(para.solver.epochs,1).fill_(0)


if(para.model.loss == 'L2'):
    criterion = nn.MSELoss()
elif (para.model.loss == 'L1'):
    criterion = nn.L1Loss()
if(para.model.optimizer == 'Adam'):
    optimizer = optim.Adam(net.parameters(), lr= para.solver.lr)
elif (para.model.optimizer == 'SGD'):
    optimizer = optim.SGD(net.parameters(), lr= para.solver.lr, momentum=0.9)
if (para.model.scheduler == 'CosAn'):
    scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0)

optimizer_template = optim.Adam(net.parameters(), lr= para.solver.lr)
scheduler_template = CosineAnnealingLR(optimizer_template, T_max=len(trainloader), eta_min=0)



########### Augmentation Training ###########
total_loss = []

print ('*****Training Starts*****')

for epoch in trange(para.solver.epochs): 
    total= 0; 
    total_val = 0; 
    latent_f = []
    net.train()
    count = 0
    
    for j, image_data in enumerate(trainloader):
        inputs, batch_labels = image_data
        inputs = inputs.to(dev)
        b, c, w, h = inputs.shape
        src_bch = inputs[:,0,...].reshape(b,1,w,h)
        tar_bch = inputs[:,1,...].reshape(b,1,w,h)
        
        optimizer.zero_grad()
        pred = net(src_bch, tar_bch, registration = True)     
        loss = criterion(pred[0], tar_bch) 
        loss2 = loss_Reg(pred[1])
        loss_total = loss + 0.2*loss2 + 1e-7 * pred[3]
        loss_total.backward(retain_graph=True)
        optimizer.step()
        
        running_loss += loss_total.item()
        total += running_loss
        running_loss = 0.0
        count += 1
    
    total_loss.append(total/count)
    
    
PATH = './pre_trained_models/MGAug_2DShape.pth'
torch.save(net, PATH)

print ('*****Training Ends*****')
print ('Model Saved at: ', PATH)
