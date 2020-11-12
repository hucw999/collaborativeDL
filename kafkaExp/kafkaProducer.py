from kafka import KafkaProducer
import time
from PIL import Image
import pickle
producer = KafkaProducer(bootstrap_servers='10.4.10.239:9092')  # 连接kafka


# msg = [1,2,3,4]
for i in range(20):
    img = Image.open('../imgs/test.jpg')
    img = pickle.dumps(img)
    # producer.send('byzantine', str(msg[i]).encode('utf-8'))  # 发送的topic为test
    producer.send('test',img)
    time.sleep(1)
producer.close()

# {\"a\":\"123123\"}