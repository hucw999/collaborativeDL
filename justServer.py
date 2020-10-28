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
# IP=socket.gethostname()
from kazoo.client import KazooClient
# print(IP)
# IP = "127.0.0.1"
# PORT=8501
#
# zk = KazooClient(hosts='127.0.0.1:2181')    #如果是本地那就写127.0.0.1
# zk.start()    #与zookeeper连接
# #makepath=True是递归创建,如果不加上中间那一段，就是建立一个空的节点
# node = zk.get_children('/server/ip')
# if node == None:
#     zk.create('/server/ip',b'127.0.0.1',makepath=True)
# node = zk.get_children('/server/port')
# if node == None:
#     zk.create('/server/port',b'8501',makepath=True)
#  # 查看根节点有多少个子节点
#
# print(node)
# print(zk.get('/server')[0])
# zk.stop()

class RpcData:

    def __init__(self,tdata,met):
        self.tensorData = tdata
        self.method = met

def severCompute(server):
    model = myvgg.myVgg(st=19,ed = 18 ,part=2)
    # model = myresnet.myResnet18(part=2)
    pth = "../checks/vgg16-397923af.pth"

    checkpoint = torch.load(pth)

    # model.load_state_dict(checkpoint)

    model.eval()

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
            data = conn.recv(4096*65536)

            # num += len(data)
            # print(len(data))
            total_data += data
            if total_data.endswith(b"$$$$"):

                break
        # print(total_data)
    #     sock.close()
    # sock.close()
    # while True:
    #     conn, addr = server.accept()
        st = time.time()
    #     data = conn.recv(4096)
    #     print(data)

        revData = pickle.loads(total_data[0:-3])

        # print('attr: ' + getattr(revData,'method'))
        #
        # revData = revData.tensorData


        print(revData)

        # 装载模型



        # output = model(torch.from_numpy(revData))
        output = model(revData)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print(pred)

        sendData = pickle.dumps(pred)
        conn.send(sendData)
        ed = time.time()
        print('time spent:', ed - st)


if __name__=="__main__":



    server=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.setblocking(1)
    server.bind(("127.0.0.1", 8502))
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

