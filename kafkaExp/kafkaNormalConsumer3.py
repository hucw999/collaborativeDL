from kafka import KafkaConsumer,KafkaProducer
import time
import threading
from PIL import Image
from MyConsumer import *

consumer = KafkaConsumer('testyg',
                 group_id="test_group_1",
                 bootstrap_servers=['10.4.10.239:9092'],
                 enable_auto_commit=False)
producer = KafkaProducer(bootstrap_servers='10.4.10.239:9092')  # 连接kafka
clintInf = ClientInf(0,18)
#
# img = Image.open('../imgs/test.jpg')
#
# input = clintInf.transformData(input)
# torch.unsqueeze(input, dim=0).float()
# clintInf.inf(input)
cnt = 0
for msg in consumer:
    #     recv = "%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value)
    #
    #     print(recv)
    try:
        # 轮询一个batch 手动提交一次
        # print(msg.value)
        print('get msg')
        cnt += 1
        print(cnt)
        input = pickle.loads(msg.value)
        input = clintInf.transformData(input)
        input = torch.unsqueeze(input, dim=0).float()
        output = clintInf.inf(input)
        print(output)
        time.sleep(1)
        # print(self.name + ' consumed ' + str(msg.value))
        consumer.commit()  # 提交当前批次最新的偏移量. 会阻塞  执行完后才会下一轮poll
        producer.send('result', b'1',b'10.4.10.214')
        # if cnt %5 == 0:
        #     consumer.unsubscribe()
        #     consumer.subscribe('testyg')
        #     print(consumer.assignment())

    except Exception as e:

        print('commit failed', str(e))






