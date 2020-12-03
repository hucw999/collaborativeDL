import socket  # 导入 socket 模块
import argparse
import numpy as np
import os
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
import cv2
# 加载模型
from PIL import Image
from log import *
from conf.getConf import *
import json
from ZK.GetDeviceInfo import getColDevice

pwd = os.path.dirname(__file__)
host,port = getColServer()
print(host,port)
log = KafkaLog()

with open(pwd + '/imgs/label/imagenetLabel.json') as f:
    labels = json.load(f)


def class_id_to_label(i):
    return labels[i]

def inf():
    # soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # print(soft, hard)
    # resource.setrlimit(resource.RLIMIT_AS, ((4 *(1024**3))  , hard))



    model = myvgg.myVgg(part=1, st=0, ed = 18)
    # model = myresnet.myResnet18(part=1)

    pth = pwd + "/ftp/test/vgg16.pth"

    checkpoint = torch.load(pth)

    model.load_state_dict(checkpoint, strict=False)


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
            return dats,targets


        def inf(self,dat):
            # 装载模型

            with torch.no_grad():
            # 推理

                model.eval()
                output = model(dat)

            print(output.shape)
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
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])



    img = Image.open('imgs/test.jpg')

    cv2Img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    img = val_transforms(img)

    img = torch.unsqueeze(img, dim=0).float()
    # label = inf.inf(img)
    # print(label.shape)


    # print(output.shape)
    #
    proxy = Proxy(inf)
    st = time.time()

    log.logSend("DATAFLOW " + getLocalhost() +" send data to " + host)
    label = proxy.inf(img)

    label = class_id_to_label(label)
    log.logSend("INFO " + host + " get result "+ label)

    node = {
        "longitude": 116.201929,
        "latitude": 39.275255,
    }

    node = json.dumps(node)
    print(node)
    notification = {"longitude": "120.2019",
                    "latitude": "30.275255",
                    "ip": '10.4.10.123',
                    "category": "xxx",
                    "captureNode": node,
                    "targets": [label],
                    # "date": time.time()

                    }

    notification = json.dumps(notification).encode()
    from kafka import KafkaProducer, KafkaConsumer
    producer = KafkaProducer(bootstrap_servers='10.4.10.254:9092')

    producer.send("gateway", notification)

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cv2Img, label, (50,50),
                FONT, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('img',cv2Img)
    key2 = cv2.waitKey(0)

    # ed = time.time()
    # print('time spent:' ,ed-st)


if __name__ == '__main__':
    getColDevice()
    inf()

