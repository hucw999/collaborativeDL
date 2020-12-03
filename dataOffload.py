from kafka import KafkaProducer,KafkaAdminClient,KafkaConsumer
import time
from PIL import Image
import pickle
import redis
import json
from MyConsumer import *
from log import KafkaLog
import configparser
from conf.getConf import *

pwd = os.path.dirname(__file__)

kafkaHosts = getKafkaHosts()
localhost = getLocalhost()
zkHosts = conf.get('zk.server','hosts')
redisHost = conf.get('redis.server','host')
redisPort = conf.get('redis.server','port')

log = KafkaLog()

zk = KazooClient(hosts=zkHosts)
zk.start()

consumer = KafkaConsumer('result',
                 group_id="test_group_1",
                 bootstrap_servers=[kafkaHosts]
                         )

conn = redis.Redis(redisHost, port=redisPort)


def dataPub():

    cnt = 0


    producer = KafkaProducer(bootstrap_servers='10.4.10.239:9092')  # 连接kafka
    devNum = len(producer.partitions_for('testyg'))


    print(producer.metrics())

    st= time.time()
    for i in range(30):
        print(i)
        # if i%2 == 0:
        img = Image.open(pwd + 'imgs/test.jpg')
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
    log.logSend("INFO "+ localhost + " publish 30 msgs!")

    colDevs = []
    for msg in consumer:
        print(cnt)
        # print(msg)
        # if cnt == 0:
            # st = time.time()
        cnt += 1
        if colDevs.count(msg.key.decode()) == 0:
            colDevs.append(msg.key.decode())
        if cnt == 30:
            # ed =time.time()
            # print(ed-st)
            ed = time.time()
            consumer.close()
    costTime= ed -st

    log.logSend("INFO" + localhost
                + "'s datas handle done, costs time " + costTime)



    taskInfo = {
                'name':'dataOffload',
                'type': 'classification',
                'startDevice': localhost,
                'dataNum': 30,
                'devNum': devNum,
                'colDevs':colDevs,
                'latency': ed-st
                }
    js = json.dumps(taskInfo)

    conn.lpush("kafkaTasks",js)

    producer.close()
    # {\"a\":\"123123\"}

if __name__ == "__main__":
    dataPub()