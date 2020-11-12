import socket  # 导入 socket 模块
# import redis
import argparse
import numpy as np
import os
from MyConsumer import *
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
from PIL import Image
# 加载模型

def inf(model, dat):



    st = time.time()
    with torch.no_grad():
        # 推理

        model.eval()


        # data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)


        output = model(dat)
    ed = time.time()
    print('time spent:', ed - st)

if __name__ == "__main__":
    clintInf = ClientInf(0,18)
    clintInf.loadModel()
    st = time.time()
    for i in range(20):


    # for filename in os.listdir(r"imgs/"):
        img = Image.open('imgs/' + 'test.jpg')


        # if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        #     img = Image.open('imgs/' + filename)
        # else:
        #     continue
        img = clintInf.transformData(img)

        input = torch.unsqueeze(img, dim=0).float()
        output = clintInf.inf(input)
        print(output)
    ed = time.time()
    print('time spent:', ed - st)