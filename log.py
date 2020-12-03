from kazoo.client import KazooClient
import json
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import configparser
from conf import getConf

from kafka import KafkaProducer


class KafkaLog:
    def __init__(self):


        kafkaHost = getConf.getKafkaHosts()

        self.producer = KafkaProducer(bootstrap_servers=kafkaHost)  # 连接kafka

        self.consoleChanel = getConf.getConsoleTopic()

    def logSend(self, msg):
        try:
            self.producer.send(self.consoleChanel, msg.encode())
        except Exception as e:
            print("send msg to kafka fail! ", e)