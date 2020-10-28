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


class RpcData:

    def __init__(self,tdata,met):
        self.tensorData = tdata
        self.method = met

def severCompute(server):
    model = myvgg.myVgg(st=2,ed = 18 ,part=1)
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

        st = time.time()


        revData = pickle.loads(total_data[0:-3])

        output = model(revData)



        s = socket.socket()  # 创建 socket 对象
        # host = socket.gethostname()  # 获取本地主机名
        # host = "127.0.0.1"
        # port = 8501  # 设置端口号

        s.connect(("127.0.0.1", 8502))

        msg = pickle.dumps(output)
        msg += b"$$$$"
        print(msg)
        s.send(msg)

        pred = s.recv(1024)

        # sendData = pickle.dumps(pred)
        conn.send(pred)
        ed = time.time()
        print('time spent:', ed - st)
    s.close()

    conn.close()

if __name__=="__main__":



    server=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.setblocking(1)
    server.bind(("127.0.0.1", 8501))
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

