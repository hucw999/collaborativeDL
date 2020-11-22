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

from kafka import KafkaProducer
from ftp.ftpClient import FTPClient

def selectModel():
    # 配置信息
    conf = configparser.ConfigParser()
    conf.read('/home/huchuanwen/bishe/collaborativeDL/conf/inf.conf')

    kafkaHost = conf.get("ssd.kafkaServer", "hosts")

    localhost = conf.get('ssd.local', 'host')

    producer = KafkaProducer(bootstrap_servers=kafkaHost)  # 连接kafka

    consoleChanel = conf.get("ssd.kafkaServer", "consoleChanel")
    zk = KazooClient(hosts="10.4.10.239:2181")
    zk.start()


    p = zk.get('/selectModel/param/p')


    # print(p)
    producer.send(consoleChanel, ("INFO device " + localhost + " get model selector parameters").encode())

    p = pickle.loads(p[0])

    best_predicted_arm = np.argmax(p)

    if best_predicted_arm == 0:
        ftp = FTPClient.ftpconnect("127.0.0.1", "ftp***", "Guest***")
        FTPClient.downloadfile(ftp, "vgg16-397923af.pth", "/home/huchuanwen/bishe/collaborativeDL/ftp/test/vgg16.pth")

    print(best_predicted_arm)

    producer.send(consoleChanel, ("INFO device " + localhost + " select model vgg16").encode())

    zk.stop()

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