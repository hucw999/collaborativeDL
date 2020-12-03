
from MyConsumer import *

import socket
import argparse
import numpy as np
import os
import json
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

import resource


def soloinf():
    # soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # print(soft, hard)
    # resource.setrlimit(resource.RLIMIT_AS, ((2 * (1024 ** 3)), hard))

    model = myvgg.myVgg(part=0, st=0, ed=18)
    # model = myresnet.myResnet18(part=1)

    pth = "/home/huchuanwen/bishe/checks/vgg16-397923af.pth"

    checkpoint = torch.load(pth)


    model.load_state_dict(checkpoint, strict=False)
    checkpoint = None


    # torchvision.models.resnet18()

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
            dat = datasets.CIFAR10(
                '/Users/huchuanwen/hcw/graduate/rethinking-network-pruning/cifar/l1-norm-pruning/data.cifar10',
                train=True, download=True,
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
            # testData = np.transpose(dat.data[0], (2, 0, 1))
            # testData = torch.from_numpy(testData).float()
            # testData = torch.unsqueeze(testData, dim=0)

        def inf(self, dat):
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

                pred = pickle.loads(recvData)

                print(pred)

                s.close()
                ed = time.time()
                print('transfer time spent:', ed - st)

                print("after print")

                return pred

            return newAttr


    inf = ClientInf()

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    img = Image.open('/home/huchuanwen/bishe/collaborativeDL/imgs/test.jpg')

    img = val_transforms(img)

    img = torch.unsqueeze(img, dim=0).float()
    label = inf.inf(img)
    print(label.shape)








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
        time.sleep(1)
        print(output)
    ed = time.time()
    print('time spent:', ed - st)

    import redis

    conn = redis.Redis(host='10.4.10.228', port=6379)
    taskInfo = {
                'name':'solo',
                'type': 'classification',
                'startDevice': '10.4.10.194',
                'dataNum': 20,
                'devNum':1,
                'latency': ed-st
                }
    js = json.dumps(taskInfo)

    conn.lpush("kafkaTasks",js)