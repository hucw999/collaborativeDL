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
import pickle
import time
from models import *


class ClientInf:

    rdb = None

    cacheResults = {}

    def getData(self):
        # 加载数据
        dat = datasets.CIFAR10('./data.cifar10', train=True, download=True,
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
        # testData = np.transpose(dat.data[0], (2, 0, 1))
        # testData = torch.from_numpy(testData).float()
        # testData = torch.unsqueeze(testData, dim=0)


    def inf(self,dat):
        # 装载模型
        model = vgg(dataset='cifar10', depth=16, part=1)
        pth = 'logs/model_best.pth.tar'
        print("=> loading checkpoint '{}'".format(pth))
        checkpoint = torch.load(pth)

        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])


        with torch.no_grad():
        # 推理
            st = time.time()
            model.eval()
            ed = time.time()
            print('time spent:', ed - st)

        # data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)

            output = model(dat)

        print(output.shape)
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # print(pred)
        return output

    def readRedis(self):
        keys = self.rdb.keys("*")
        for key in keys:
            # k = pickle.loads(key)
            self.cacheResults[key] = self.rdb.get(key)
        print(self.cacheResults)

    def connectRedis(self):
        self.rdb = redis.StrictRedis(host='localhost', port=6379, db=0)

    def writeRedis(self,input,label):
        self.rdb.set(input,label)


# RPC远程调用


class RpcData:

    def __init__(self, tdata, met):
        self.tensorData = tdata
        self.method = met


class Proxy(object):

    def __init__(self, target):
        self.target = target

    def __getattribute__(self, name):
        target = object.__getattribute__(self, "target")
        attr = object.__getattribute__(target, name)

        print('name:' + name)

        def newAttr(*args, **kwargs):  # 包装
            print("before print")

            out = attr(*args, **kwargs)

            sendData = pickle.dumps(out)

            s = socket.socket()  # 创建 socket 对象
            host = socket.gethostname()  # 获取本地主机名
            port = 12345  # 设置端口号

            s.connect((host, port))

            s.send(sendData)

            # s.send('12'.encode())

            recvData = s.recv(1024)

            print(recvData)

            pred = pickle.loads(recvData)

            print(pred)

            s.close()

            print("after print")

            return pred

        return newAttr
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

inf = ClientInf()

dat = inf.getData()
st = time.time()
inf.inf(dat)
ed = time.time()
print('time spent:' ,ed-st)
# inf.connectRedis()
#
# # inf.rdb.flushdb()
#
# inf.readRedis()
# dat = inf.getData()
# inf.inf(dat)
# datas,targets = inf.getDatas()
# for data in datas:
#     output = inf.inf(data).numpy()
#     saveData = pickle.dumps(output)
#     inf.writeRedis(saveData,"xxx")

# proxy = Proxy(inf)
# proxy.inf()


# dat = inf.getData()
# output = inf.inf(dat).numpy()
# for key in inf.cacheResults.keys():
#     kOut = pickle.loads(key)
#     print(cos_sim(output,kOut))
#     if cos_sim(output,kOut) > 0.97:
#         # k = pickle.dumps(key)
#         # print(k)
#         # inf.rdb.set(key,"123")
#
#
#         print(inf.rdb.get(key))
#         break
#         # print(inf.rdb.get(k))
#         # print(inf.cacheResults[key])




# test cache speed

# cache = []
# outputs = inf.inf().numpy()
# for i in range (10):
#     cache.append(outputs)
#
#
# st = time.time()
# for cacheOutput in cache:
#
#     cos_sim(cacheOutput,outputs)
# ed = time.time()
# print('time spent:' ,ed-st)