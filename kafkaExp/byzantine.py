from kafka import KafkaConsumer
import time
from kafka import TopicPartition


partiton0 = TopicPartition('byzantine', 0)
partiton1 = TopicPartition('byzantine', 1)
consumer = KafkaConsumer(
                         group_id="test_group_2",
                         bootstrap_servers=['10.4.10.239:9092'],
                         enable_auto_commit=False)

consumer.assign(partitions=[partiton0])
f = 0
for msg in consumer:
#     recv = "%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value)
#
#     print(recv)
    try:
          # 轮询一个batch 手动提交一次
        time.sleep(3)
        print('attacked '+ str(msg.value) + 'partition is ' + str(msg.partition))
        consumer.commit()  # 提交当前批次最新的偏移量. 会阻塞  执行完后才会下一轮poll
        if f == -1:
            consumer.assign(partitions=[partiton0])
        else:
            consumer.assign(partitions=[partiton1])
        f = ~f
        print(f)

    except Exception as e:

        print('commit failed', str(e))