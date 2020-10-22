import socket  # 导入 socket 模块
import redis
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
# 加载模型


def getData():
    # 加载数据
    dat = datasets.CIFAR10(
        '/Users/huchuanwen/hcw/graduate/rethinking-network-pruning/cifar/l1-norm-pruning/data.cifar10', train=True,
        download=True,
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

def getDatas(self, num=10):
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

    return dats, targets

model = vgg(dataset='cifar10', depth=16, part=0)
pth = 'logs/model_best.pth.tar'
print("=> loading checkpoint '{}'".format(pth))
checkpoint = torch.load(pth)

best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
# 单机


dat = getData()
st = time.time()
with torch.no_grad():
    # 推理

    model.eval()


    # data, target = data.cuda(), target.cuda()
    # data, target = Variable(data, volatile=True), Variable(target)

    output = model(dat)
ed = time.time()
print('time spent:', ed - st)