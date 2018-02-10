#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 02:10:27 2018

@author: soham
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import os
import csv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(6,128)
        self.fc2=nn.Linear(128,4)
    def forward(self,x):
        x=F.sigmoid(self.fc1(x))
        x=F.dropout(x)
        #x=F.sigmoid(self.fc2(x))
        x=self.fc2(x)
        return x
net =Net()
net = net.cuda()
criterion = nn.NLLLoss(size_average=True)
learn_rate=0.001

print (net)
optimizer = optim.Adam(net.parameters(),lr=0.0001)
rows=[]
with open('train.csv','r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    itr=0
    for row in spamreader:
        if itr==0:
            itr+=1
            continue
        else:
            rows.append(row[0].split(','))
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            rows[i][j] = int(rows[i][j])

rows = np.array(rows)
inputs = rows[:,0:6]
labels = rows[:,6]
inputs = Variable(torch.from_numpy(inputs)).float().cuda()
labels = (Variable(torch.from_numpy(labels)-1)).cuda()
best = 0.0
for ep in range(50000):
    outputs = net(inputs)
    loss = criterion(F.log_softmax(outputs,dim=1), labels)
    _, predicted = torch.max(outputs.data, 1)
    loss.backward()
    optimizer.step()
    success = 0
    total = 0
    if ep%100 == 0:
        for i in range(len(labels)):
            if ( predicted[i] == labels[i].data[0]):
                success = success+1
                total = total + 1
            else:
                total = total +1
        if (success/float(total)>best):
            best = success/float(total)
            print (ep,best)
            torch.save(net,'model.pth')




