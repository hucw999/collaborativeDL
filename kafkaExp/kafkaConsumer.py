from kafka import KafkaConsumer
import time
import threading

consumer1 = KafkaConsumer('byzantine',
                         group_id="test_group_1",
                         bootstrap_servers=['10.4.10.239:9092'],
                         enable_auto_commit=False)
consumer2 = KafkaConsumer('byzantine',
                         group_id="test_group_0",
                         bootstrap_servers=['10.4.10.239:9092'],
                         enable_auto_commit=False)
# while True:
#     msg = consumer.poll(max_records = 1, timeout_ms=60000)
#     try:
#           # 轮询一个batch 手动提交一次
#         time.sleep(3)
#         print(msg.items())
#         consumer.commit()  # 提交当前批次最新的偏移量. 会阻塞  执行完后才会下一轮poll
#
#
#     except Exception as e:
#
#         print('commit failed', str(e))
#     # print(msg)

class myThread (threading.Thread):
    def __init__(self, threadID, name,  consumer):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.consumer = consumer
    def run(self):
        print ("开始线程：" + self.name)
        for msg in self.consumer:
            #     recv = "%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value)
            #
            #     print(recv)
            try:
                # 轮询一个batch 手动提交一次
                time.sleep(3)
                print(self.name + ' consumed ' + str(msg.value))
                self.consumer.commit()  # 提交当前批次最新的偏移量. 会阻塞  执行完后才会下一轮poll


            except Exception as e:

                print('commit failed', str(e))
        print ("退出线程：" + self.name)




t1 = myThread(1,'t1',consumer1)
t1.start()


t2 = myThread(2,'t2',consumer2)
t2.start()