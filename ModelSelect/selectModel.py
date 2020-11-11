import ftp.ftpClient as ftp
import numpy as np
import pickle
from kazoo.client import KazooClient
import json
import torch
from torchvision import transforms
from PIL import Image
import torchvision

zk = KazooClient(hosts="10.4.10.239:2181")
zk.start()


p = zk.get('/selectModel/param/p')
# print(p)

p = pickle.loads(p[0])

best_predicted_arm = np.argmax(p)

print(best_predicted_arm)

vgg = torchvision.models.vgg16()
pth = "../ftp/test/vgg16.pth"

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

index = np.argmax(output)

with open('../imgs/imagenetLabel.json') as f:
    labels = json.load(f)


def class_id_to_label(i):
    return labels[i]


print(class_id_to_label(index))
# print(output)

zk.stop()