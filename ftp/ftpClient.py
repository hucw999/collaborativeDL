from ftplib import FTP
import time
import tarfile
import os
import torchvision
import torch
from PIL import Image
from torchvision import datasets, transforms
import numpy as np

# !/usr/bin/python
# -*- coding: utf-8 -*-
from ftplib import FTP
def ftpconnect(host, username, password):
  ftp = FTP()
  # ftp.set_debuglevel(2)
  ftp.connect(host, 2121)
  # ftp.login(username, password)
  ftp.login(user = 'anonymous', passwd ='')
  return ftp
#从ftp下载文件
def downloadfile(ftp, remotepath, localpath):
  bufsize = 1024
  fp = open(localpath, 'wb')
  ftp.retrbinary('RETR ' + remotepath, fp.write, bufsize)
  ftp.set_debuglevel(0)
  fp.close()
#从本地上传文件到ftp
def uploadfile(ftp, remotepath, localpath):
  bufsize = 1024
  fp = open(localpath, 'rb')
  ftp.storbinary('STOR ' + remotepath, fp, bufsize)
  ftp.set_debuglevel(0)
  fp.close()
if __name__ == "__main__":
  ftp = ftpconnect("127.0.0.1", "ftp***", "Guest***")
  downloadfile(ftp, "vgg16-397923af.pth", "./test/vgg16.pth")

  vgg = torchvision.models.vgg16()
  pth = "./test/vgg16.pth"

  checkpoint = torch.load(pth)

  vgg.load_state_dict(checkpoint)

  vgg.eval()

  img = Image.open('../imgs/test.jpg')

  loader = transforms.Compose([
    transforms.ToTensor()])

  img = loader(img).unsqueeze(0)
  # img = val_transforms(img)

  # img = torch.unsqueeze(img, dim=0).float()
  with torch.no_grad():
    output = vgg(img).numpy()

  output = np.argmax(output)

  print(output)