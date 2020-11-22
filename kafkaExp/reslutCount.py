from kafka import KafkaConsumer
import time
import threading
from PIL import Image
from MyConsumer import *

consumer = KafkaConsumer('result',
                 group_id="test_group_1",
                 bootstrap_servers=['10.4.10.239:9092']
                         )
cnt = 0
for msg in consumer:
    print(cnt)
    # print(msg)
    if cnt == 0:
        st = time.time()
    cnt += 1
    if cnt == 20:
        ed =time.time()
        print(ed-st)