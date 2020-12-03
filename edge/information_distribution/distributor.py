import datetime
import json
import os
import threading
import time
from subprocess import Popen

from typing import Union, List, Callable

from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.producer.future import FutureRecordMetadata

from edge.information_distribution.utils import generate_random_str
# from producer import Producer
from edge.speed_limit.limiter import SpeedLimiter
from edge.utils import *

pwd = os.path.dirname(__file__)

DISTRIBUTE_TOPIC = 'location_changed'
TRIGGER_TOPIC = 'trigger_location_change'
RECEIVED_TOPIC = 'location_changed_received'
LIMIT_SPEED_TOPIC = 'limit_speed'
START_CAPTURE_TOPIC = 'start_capture'
CONSOLE_TOPIC = 'console'
RPC_TOPIC = 'rpc'

localhost = get_local_ip()

class Distributor:
    def __init__(self, name: str, trigger: Callable[[any], None],
                 kafka_bootstrap_servers: Union[str, List[str]] = '10.4.10.239:9092'):
        # self.parent = parent
        self.name = name
        self.consumer = KafkaConsumer(DISTRIBUTE_TOPIC, TRIGGER_TOPIC, LIMIT_SPEED_TOPIC, START_CAPTURE_TOPIC,RPC_TOPIC,
                                      bootstrap_servers=kafka_bootstrap_servers, value_deserializer=json.loads)
        self.consuming_thread = threading.Thread(target=self._keep_consuming)
        self.producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers)
        self.trigger = trigger
        self.limiter = SpeedLimiter()
        self.limiter.current_limit = None
        self.consuming_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.consumer.close()
        self.producer.close()

    def _keep_consuming(self):
        print("consuming")
        msg: ConsumerRecord
        print(self.consumer)
        for msg in self.consumer:
            value = msg.value
            print(value)
            if msg.topic == TRIGGER_TOPIC:
                if self.name == value['targetName']:
                    self.trigger(msg.value)
            elif msg.topic == DISTRIBUTE_TOPIC:
                now = time.time()
                self.producer.send(RECEIVED_TOPIC,
                                   json.dumps(
                                       {'name': self.name, 'send_time': value['send_time'],
                                        'receive_time': now}).encode())
            elif msg.topic == LIMIT_SPEED_TOPIC:
                new_limit = value['limit']
                self.limiter.current_limit = new_limit
                # if new_limit == 'high':
                #     self.parent.set_max_bandwidth(20)
                # elif new_limit == 'low':
                #     self.parent.set_max_bandwidth(0.019)
                # else:
                #     self.parent.set_max_bandwidth(100)
                # print('new limit: ' + (new_limit or 'None'))
            elif msg.topic == START_CAPTURE_TOPIC:
                if self.name == value['captureNode']:
                    distributed_dnn_path = os.path.expanduser('~/Develop/distributed-DNN/')
                    Popen(['python3', 'demo/liveClient.py', value['uuid'], value['captureNode'],
                           value['calculationNode2'],
                           value['calculationNode2Ip']],
                          cwd=distributed_dnn_path)
            elif msg.topic == RPC_TOPIC:
                print(value)
                if value['host'] == localhost:
                    print(value['host'])
                    if value['func'] == 'startServer':
                        print(value['func'])
                        # thread = threading.Thread(target=liveServer.startLiveServer())
                        # thread.start()
                        Popen(['python3.6', pwd + '/../../justServer.py'])
                    elif value['func'] == 'startInf':
                        print(value['func'])
                        Popen(['python3.6', pwd + '/../../justClient.py'])
                        # Popen(['sudo', 'cgexec', '-g' ,'memory:memoryInf' ,'python3.6', '/home/huchuanwen/bishe/collaborativeDL/solo.py'])
                    elif value['func'] == 'selectModel':
                        print(value['func'])
                        Popen(['python3.6', '/../../selectModel.py'])
                    elif value['func'] == 'dataOffload':
                        print(value['func'])
                        Popen(['python3.6', pwd + '/../../dataOffload.py'])
                    elif value['func'] == 'dataConsume':
                        print(value['func'])
                        Popen(['python3.6', pwd + '/../../dataConsume.py'])

    def distribute(self, latitude: float, longitude: float) -> FutureRecordMetadata:
        msg_dict = {
            'name': self.name,
            'send_time': datetime.datetime.now().isoformat(),
            'lat': latitude,
            'long': longitude,
            'msg': 'dangerous! Found enemy_' + generate_random_str(105)
        }
        msg_dict_str = json.dumps(msg_dict)
        # should extend to 128 bytes
        # rest_len = 128 - len(msg_dict_str)
        # if rest_len > 0:
        #     msg_dict['msg'] = generate_random_str(128)
        #     msg_dict_str = json.dumps(msg_dict)
        return self.producer.send(DISTRIBUTE_TOPIC, msg_dict_str.encode())
