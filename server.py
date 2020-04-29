import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import socket
import threading
# import pickle
import io
import sys
import time
from models import *
import _pickle as pickle
IP=socket.gethostname()
# IP = "127.0.0.1"
PORT=8501


class RpcData:

    def __init__(self,tdata,met):
        self.tensorData = tdata
        self.method = met

def severCompute(server):
    while True:
        conn, addr = server.accept()
        data = conn.recv(4096)
        print(data)

        revData = pickle.loads(data)

        # print('attr: ' + getattr(revData,'method'))
        #
        # revData = revData.tensorData


        print(revData)

        # 装载模型
        model = vgg(dataset='cifar10', depth=16, part=2)
        pth = 'logs/model_best.pth.tar'
        print("=> loading checkpoint '{}'".format(pth))
        checkpoint = torch.load(pth)

        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])

        model.eval()
        # output = model(torch.from_numpy(revData))
        output = model(revData)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print(pred)

        sendData = pickle.dumps(pred)
        conn.send(sendData)


if __name__=="__main__":



    server=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setblocking(1)
    server.bind((IP, PORT))
    print("云端启动，准备接受任务")
    server.listen(5)
    severCompute(server)

# while True:
#
#     try:
#         receiveData(server)
#     except BlockingIOError:
#         pass
#     finally:
#         pass

