import socket  # 导入 socket 模块
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
import _pickle as pickle
import sys
import torchvision
from models import *
from kazoo.client import KazooClient
# 加载模型
from PIL import Image
from cache.MyCache import LRUCache
import redis
from conf.getConf import *


localhost = getLocalhost()
model = myvgg.myVgg(part=1,st=0,ed = 18)
# model = myresnet.myResnet18(part=1)
pwd = os.path.dirname(__file__)
pth = pwd + "/../ftp/test/vgg16.pth"

checkpoint = torch.load(pth)

model.load_state_dict(checkpoint, strict = False)

# torchvision.models.resnet18()

host,port = getColServer()

redisHost = conf.get('redis.server','host')
redisPort = conf.get('redis.server','port')
conn = redis.Redis(redisHost, port=redisPort)

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
            # pred = output.data.max(1, keepdim=True)[1]



        return output

    def colInf(self,dat):
        return dat


import json

with open(pwd + '/../imgs/label/imagenetLabel.json') as f:
    labels = json.load(f)


def class_id_to_label(i):
    return labels[i]


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
            # print("before print")
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

            recvData = s.recv(1024)



            pred = pickle.loads(recvData)
            # pred = pred.data.max(1, keepdim=True)[1]

            print(pred)


            s.close()
            ed = time.time()
            print('transfer time spent:', ed - st)

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


if __name__ == "__main__":

    inf = ClientInf()



    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])



    # img = val_transforms(img)

    proxy = Proxy(inf)
    imgs = []
    cache = LRUCache(3)
    for filename in os.listdir(pwd+"/../imgs/"):

        st = time.time()
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = Image.open(pwd+'/../imgs/' + filename)
        else:
            continue
        # imgs.append(val_transforms(img))

        img = val_transforms(img)


        input = torch.unsqueeze(img, dim=0).float()
        output = inf.inf(input)



        outKey = torch.reshape(output,(1,512*7*7))
        # print(output.shape)

        label = cache.get_sim(outKey)

        if label == -1:
            label = proxy.colInf(output)
            label = class_id_to_label(label)
            print(label)
            ed =time.time()
            print('no cache ', ed-st)
            cache.put(outKey, label)
            taskInfo = {
                'name': localhost+":"+filename,
                'type': 'col',
                'startDevice': localhost,
                'colDevice': host,
                'cacheHit': False,
                'latency': ed - st,
                'target': label
            }
            js = json.dumps(taskInfo)

            conn.lpush("dnnTasks", js)
        else:
            ed = time.time()
            # label = class_id_to_label(label)
            print(label)
            print('with cache ',ed-st)

            taskInfo = {
                'name': localhost+":"+filename,
                'type': 'col',
                'startDevice': localhost,
                'colDevice': host,
                'cacheHit': True,
                'latency': ed - st,
                'target': label
            }
            js = json.dumps(taskInfo)

            conn.lpush("dnnTasks", js)


    # proxy = Proxy(inf)
    # st = time.time()
    #
    #
    # proxy.inf(img)
    # ed = time.time()
    # print('time spent:' ,ed-st)




