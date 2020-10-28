from kafka import KafkaProducer
import time
from PIL import Image
import pickle
producer = KafkaProducer(bootstrap_servers='10.4.10.239:9092')  # 连接kafka

img = Image.open('../imgs/test.jpg')
img = pickle.dumps(img)
msg = [1,2,3,4]
for i in range(3):
    producer.send('byzantine', str(msg[i]).encode('utf-8'))  # 发送的topic为test
producer.close()
# {\"a\":\"123123\"}