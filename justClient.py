import socket  # 导入 socket 模块
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
import torchvision
from models import *
from kazoo.client import KazooClient
# 加载模型
from PIL import Image
model = myvgg.myVgg(part=1,st=0,ed = 18)
# model = myresnet.myResnet18(part=1)

pth = "../checks/vgg16-397923af.pth"

checkpoint = torch.load(pth)

# model.load_state_dict(checkpoint)

torchvision.models.resnet18()

host = "127.0.0.1"
port = 8502
# # 获取server
#
# zk = KazooClient(hosts='127.0.0.1:2181')    #如果是本地那就写127.0.0.1
# zk.start()    #与zookeeper连接
# #makepath=True是递归创建,如果不加上中间那一段，就是建立一个空的节点
# host = (str(zk.get('/server/ip')[0],encoding = "utf-8"))
# port = int(str(zk.get('/server/port')[0],encoding = "utf-8"))
#
# zk.stop()


class ClientInf:

    rdb = None


    cacheResults = {}


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
        # testData = np.transpose(dat.data[0], (2, 0, 1))
        # testData = torch.from_numpy(testData).float()
        # testData = torch.unsqueeze(testData, dim=0)


    def inf(self,dat):
        # 装载模型


        with torch.no_grad():
        # 推理



            model.eval()
            output = model(dat)




        # print(sys.getsizeof(output))
        # print(sys.getsizeof(output.numpy()))
        # print(output.shape)
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # print(pred)

        return output




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
            st = time.time()
            out = attr(*args, **kwargs)


            sendData = pickle.dumps(out)
            # print(sendData)
            s = socket.socket()  # 创建 socket 对象
            # host = socket.gethostname()  # 获取本地主机名
            # host = "127.0.0.1"
            # port = 8501  # 设置端口号

            s.connect((host, port))

            sendData += b"$$$$"

            s.send(sendData)

            time.sleep(10)
            s.send(sendData)

            # s.send('12'.encode())
            recvData = s.recv(1024)

            recvData2 = s.recv(1024)


            pred = pickle.loads(recvData)
            pred2 = pickle.loads(recvData2)

            print(pred)
            print(pred2)

            s.close()
            ed = time.time()
            print('transfer time spent:', ed - st)

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



val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])



img = Image.open('./imgs/test.jpg')

img = val_transforms(img)

img = torch.unsqueeze(img, dim=0).float()

# output = inf.inf(img)


# print(output.shape)

proxy = Proxy(inf)
st = time.time()


proxy.inf(img)
ed = time.time()
print('time spent:' ,ed-st)




