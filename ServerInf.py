import socket  # 导入 socket 模块
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import pickle
from models import *


class ServerInf:
    def __init__(self):
        self.model = None
        return

    def serverRegist(self, deviceInfo):
        return

    def loadModel(self):
        self.model = myvgg.myVgg(st=19, ed=18, part=2)
        # model = myresnet.myResnet18(part=2)
        pth = "../checks/vgg16-397923af.pth"

        checkpoint = torch.load(pth)

        # model.load_state_dict(checkpoint)

    def colInf(self):
        model = self.model
        model.eval()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # server.setblocking(1)
        server.bind(("127.0.0.1", 8502))
        print("云端启动，准备接受任务")
        server.listen(5)

        while True:
            print("start......")
            conn, adddr = server.accept()
            total_data = b''
            num = 0
            data = conn.recv(1024)
            total_data += data
            num = len(data)
            # 如果没有数据了，读出来的data长度为0，len(data)==0
            while len(data) > 0:
                print(len(data))
                data = conn.recv(4096 * 65536)

                # num += len(data)
                # print(len(data))
                total_data += data
                if total_data.endswith(b"$$$$"):
                    break

            st = time.time()


            revData = pickle.loads(total_data[0:-3])

            # print('attr: ' + getattr(revData,'method'))
            #
            # revData = revData.tensorData

            print(revData)

            # 装载模型


            output = model(revData)

            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability


            print(pred)

            sendData = pickle.dumps(pred)
            conn.send(sendData)
            ed = time.time()
            print('time spent:', ed - st)


