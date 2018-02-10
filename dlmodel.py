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
        self.fc1=nn.Linear(6,4)
    
    def forward(self,x):
        x=self.fc1(x)
        return x


net=Net()
criterion = nn.NLLLoss(size_average=True)
learn_rate=0.01
print net.parameters()

optimizer = optim.Adam(net.parameters())
rows=[]
with open('train.csv','rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    itr=0
    for row in spamreader:
        if itr==0:
            itr+=1
            continue
        else:
            rows.append(row)

batches=[]

for i in range(0,len(rows),4):
    tempbatch=[]
    for j in range(i,min(i+4,len(rows))):
        tempbatch.append(rows[j])
    print len(tempbatch)
    batches.append(tempbatch)


for itr,batch in enumerate(batches):
    
    try:
        net.train(True)
        arr=[]
        label=[]
        for j in range(4):
            temparr=[]
            for i in range(0,11,2):
                temparr.append(int(batch[j][0][i]))
            label.append(int(batch[j][0][12]))
            arr.append(temparr)
          
            x=torch.Tensor(4,6)
            for j in range(4):
                for i in range(6):
                    x[j][i]=float(arr[j][i])
            labels=torch.LongTensor(4)
            for i in range(4):
                labels[i]=label[i]


            
            x,labels=Variable(x),Variable(labels)
            optimizer.zero_grad()
            outputs=net(x)
            loss = criterion(F.log_softmax(outputs), labels)
            _, predicted = torch.max(outputs.data, 1)
            loss.backward()
            optimizer.step()
    except:
        print "error at ",itr
            
#with open('test.csv','rb') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    itr=0
#    for row in spamreader:
#        if itr==0:
#            itr+=1
#            continue
#        else:
#            itr+=1
#            net=net.train(False)
#            arr=[]
#            for i in range(0,11,2):
#                arr.append(int(row[0][i]))
#            x=torch.Tensor(6)
#            for i in range(6):
#                x[i]=float(arr[i])
#            
#            x=Variable(x)
#            outputs=net(x)
#            _,predicted=torch.max(outputs.data,1)
#            print predicted
#            



            
            
