import ftp.ftpClient as ftp
import numpy as np
import pickle
from kazoo.client import KazooClient
import json
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import configparser
from log import KafkaLog
from kafka import KafkaProducer
from ftp.ftpClient import FTPClient
import matplotlib.pyplot as plt

# 配置信息
conf = configparser.ConfigParser()
conf.read('/home/huchuanwen/bishe/collaborativeDL/conf/inf.conf')

log = KafkaLog()

localhost = conf.get('inf.local', 'host')

zk = KazooClient(hosts="10.4.10.239:2181")
zk.start()



def selectModel():

    A = zk.get('/selectModel/param/A')
    b = zk.get('/selectModel/param/b')
    p = zk.get('/selectModel/param/p')



    log.logSend("INFO device " + localhost + " get model selector parameters")

    p = pickle.loads(p[0])



    best_predicted_arm = np.argmax(p)

    if best_predicted_arm == 0:
        ftp = FTPClient.ftpconnect("127.0.0.1", "ftp***", "Guest***")
        FTPClient.downloadfile(ftp, "vgg16-397923af.pth", "/home/huchuanwen/bishe/collaborativeDL/ftp/test/vgg16.pth")

    print(best_predicted_arm)

    log.logSend("INFO device " + localhost + " select model vgg16")

    zk.stop()

    N = p.size
    x = np.arange(N)
    label = ("VGG16","alexnet","resnet-18")
    plt.bar(x, p, width=0.5,label="models",tick_label=label)
    plt.legend()
    plt.show()

def inf():

    vgg = torchvision.models.vgg16()
    pth = "/home/huchuanwen/bishe/collaborativeDL/ftp/test/vgg16.pth"

    checkpoint = torch.load(pth)

    vgg.load_state_dict(checkpoint)

    vgg.eval()

    img = Image.open('/home/huchuanwen/bishe/collaborativeDL/imgs/test.jpg')

    loader = transforms.Compose([
        transforms.ToTensor()])

    img = loader(img).unsqueeze(0)
    # img = val_transforms(img)

    # img = torch.unsqueeze(img, dim=0).float()
    with torch.no_grad():
        output = vgg(img).numpy()

    index = np.argmax(output)

    with open('/home/huchuanwen/bishe/collaborativeDL/imgs/label/imagenetLabel.json') as f:
        labels = json.load(f)


    def class_id_to_label(i):
        return labels[i]


    print(class_id_to_label(index))
    # print(output)

if __name__ == '__main__':
    selectModel()
    inf()