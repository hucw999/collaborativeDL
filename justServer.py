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
from log import KafkaLog
from kazoo.client import KazooClient
from conf.getConf import *

pwd = os.path.dirname(__file__)
log = KafkaLog()
localhost = getLocalhost()
print(localhost)
class RpcData:

    def __init__(self,tdata,met):
        self.tensorData = tdata
        self.method = met

def severCompute(server):
    model = myvgg.myVgg(st=19,ed = 18 ,part=2)
    # model = myresnet.myResnet18(part=2)
    pth = pwd + "/ftp/test/vgg16.pth"

    checkpoint = torch.load(pth)

    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    while True:
        print("start......")
        conn, adddr = server.accept()
        print(adddr)
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

        log.logSend(localhost + " get Intermediate output and start left computing")

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
    server.bind((localhost, 8889))
    print("云端启动，准备接受任务")
    log.logSend("INFO " + localhost + " is ready to collaborative inference!")
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

