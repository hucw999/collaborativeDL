import socket  # 导入 socket 模块
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
from models import *


class ClientInf:


    def __init__(self):

        rdb = None
        cacheResults = {}
        self.model = None
        self.ServerHost = None
        self.ServerPort = None


    def loadModel(self):
        self.model = myvgg.myVgg(part=0, st=0, ed=18)

        print(self.model)
        # model = myresnet.myResnet18(part=1)

        pth = "../checks/vgg16-397923af.pth"

        checkpoint = torch.load(pth)

        self.model.load_state_dict(checkpoint,False)
        return

    def getServer(self): #sss



        return



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

        return dats,targets
        # testData = np.transpose(dat.data[0], (2, 0, 1))
        # testData = torch.from_numpy(testData).float()
        # testData = torch.unsqueeze(testData, dim=0)


    def inf(self, dat):
        # 装载模型


        with torch.no_grad():
        # 推理



            model = self.model
            model.eval()
            output = model(dat)




        # print(sys.getsizeof(output))
        # print(sys.getsizeof(output.numpy()))
        # print(output.shape)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # print(pred)

        return pred


if __name__=="__main__":
    cli = ClientInf()
    cli.loadModel()

    import cv2

    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()


    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


    img = val_transforms(Image.fromarray(frame))

    img = torch.unsqueeze(img, dim=0).float()

    index = cli.inf(img)

    import json

    with open('./imgs/imagenetLabel.json') as f:
        labels = json.load(f)


    def class_id_to_label(i):
        return labels[i]

    print(class_id_to_label(index))

    cv2.imshow('frame', frame)
    key2 = cv2.waitKey(0)