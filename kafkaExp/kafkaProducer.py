from kafka import KafkaProducer,KafkaConsumer
import time
from PIL import Image
import pickle
import redis
import json
from MyConsumer import *


consumer = KafkaConsumer('result',
                 group_id="test_group_1",
                 bootstrap_servers=['10.4.10.239:9092']
                         )
cnt = 0


producer = KafkaProducer(bootstrap_servers='10.4.10.239:9092')  # 连接kafka
devNum = len(producer.partitions_for('testyg'))


print(devNum)


print(producer.metrics())
clintInf = ClientInf(0,18)
# msg = [1,2,3,4]
st= time.time()
for i in range(30):
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
    producer.send('testyg', img, str(i).encode())
    producer.flush()
    # time.sleep(1)
print('end')

colDevs = []
for msg in consumer:
    print(cnt)
    # print(msg)
    # if cnt == 0:
        # st = time.time()
    cnt += 1
    if colDevs.count(msg.key.decode()) == 0:
        colDevs.append(msg.key.decode())
    if cnt == 20:
        # ed =time.time()
        # print(ed-st)
        ed = time.time()
        consumer.close()


conn = redis.Redis(host='10.4.10.228', port=6379)


taskInfo = {
            'name':'dataOffload',
            'type': 'classification',
            'startDevice': '10.4.10.194',
            'dataNum': 30,
            'devNum': devNum,
            'colDevs':colDevs,
            'latency': ed-st
            }
js = json.dumps(taskInfo)

conn.lpush("kafkaTasks",js)

producer.close()
# {\"a\":\"123123\"}