import socket  # 导入 socket 模块
# import redis
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import pickle
from models import *
from PIL import Image
import torchvision.models as models
# from torchstat import stat
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



import math
#
# # def cos_sim(vector_a, vector_b):
# #     """
# #     计算两个向量之间的余弦相似度
# #     :param vector_a: 向量 a
# #     :param vector_b: 向量 b
# #     :return: sim
# #     """
# #     vector_a = np.mat(vector_a)
# #     vector_b = np.mat(vector_b)
# #     num = float(vector_a * vector_b.T)
# #     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
# #     cos = num / denom
# #     sim = 0.5 + 0.5 * cos
# #     return sim
# #
# # for i in range(1,20):
# #     model = myvgg.myVgg(True,i)
# #     model.eval()
# #     input = np.random.rand(1,3,224,224)
# #     input = torch.from_numpy(input).float()
# #     st = time.time()
# #     model(input)
# #     ed = time.time()
# #     print(i ,' : ', ed-st)
#

model = myvgg.myVgg(False,16,pretrained=False,progress=True)

pth = "../checks/vgg16-397923af.pth"

checkpoint = torch.load(pth)

# best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint)

# model = torchvision.models.resnet18(True)

# stat(model,(3,32,224))

torchvision.models.vgg16()
model.eval()



val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])



img = Image.open('./imgs/test.jpg')

img = val_transforms(img)

img = torch.unsqueeze(img, dim=0).float()

# img = torch.from_numpy(img).float()


# print(img)

st1 = time.time()
out = model(img)

print(torch.argmax(out))
ed1 = time.time()
print(1 ,' : ', ed1-st1)