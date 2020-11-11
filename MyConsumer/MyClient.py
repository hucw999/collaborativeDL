import socket  # 导入 socket 模块
# import redis
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
# import pickle
import time
import _pickle as pickle
import sys
from models import *
from kazoo.client import KazooClient



class ClientInf:

    rdb = None


    cacheResults = {}

    def __init__(self, st,ed,part=0,):
        self.model = myvgg.myVgg(part=part,st=st,ed = ed)


    def getData(self):
        # 加载数据
        dat = datasets.CIFAR10('/Users/huchuanwen/hcw/graduate/rethinking-network-pruning/cifar/l1-norm-pruning/data.cifar10', train=True, download=True,
                               transform=transforms.Compose([
                                   # transforms.Pad(4),
                                   # transforms.RandomCrop(32),
                                   # transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                               ]))
        img = transforms.ToPILImage(dat.data[0])

        img = dat.data[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维

        testData = np.transpose(dat.data[0], (2, 0, 1))
        testData = torch.from_numpy(testData).float()
        testData = torch.unsqueeze(testData, dim=0)
        return testData

    def getDatas(self,num=10):
        # 加载数据

        dats = []
        targets = []
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=1, shuffle=True)

        cnt = 0
        for data, target in test_loader:

            dats.append(data)

            targets.append(target)

            cnt += 1
            if cnt >= num:
                break
            # data, target = Variable(data, volatile=True), Variable(target)

        return dats,targets



    def transformData(self, data):
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        return val_transforms(data)

    def inf(self,dat):
        # 装载模型




        with torch.no_grad():
        # 推理



            self.model.eval()
            output = self.model(dat)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

        # print(pred)

        return pred