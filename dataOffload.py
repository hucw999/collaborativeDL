from kafka import KafkaProducer
import time
from PIL import Image
import pickle
producer = KafkaProducer(bootstrap_servers='10.4.10.239:9092')  # 连接kafka

from MyConsumer import *

print(producer.metrics())
clintInf = ClientInf(0,18)
# msg = [1,2,3,4]
for i in range(20):
    print(i)
    # if i%2 == 0:
    img = Image.open('../imgs/test.jpg')
    # else:
    #     img = Image.open('../imgs/army.jpg')

    # img = clintInf.transformData(img)

    img = pickle.dumps(img)
    print(img.__len__())
    # producer.send('byzantine', str(msg[i]).encode('utf-8'))  # 发送的topic为test
    # producer.send('result', str(i).encode())
    producer.send('testyg', img)
    producer.flush()
    # time.sleep(1)
print('end')

producer.close()

# {\"a\":\"123123\"}