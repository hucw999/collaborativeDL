import datetime
import json
import random
import shutil
import time
from typing import Union, List, Optional
import os
import psutil

import requests
from kazoo.client import KazooClient
import uuid

from kazoo.retry import KazooRetry

from edge.information_distribution.distributor import Distributor
from edge.utils import *


class Producer:
    topic = 'resource'
    last_bytes = None
    last_bytes_timestamp = None
    distributor = None
    latitude = None
    longitude = None
    max_bandwidth = 100

    def __init__(self, name: str = None, server_host: str = get_host_ip() + ':2181', activate_distributor: bool = True,
                 distributor_kafka_host: Union[str, List[str]] = get_host_ip() + ':9092'):
        self.is_virtual = False
        self.randomize_location()
        self.latency = random.uniform(0, 10)
        try:
            from jtop import jtop
            self.jetson = jtop()
            self.jetson.open()
        except Exception:
            self.jetson = None
        if 'virtual' in os.environ:
            self.is_virtual = True
        if name is None:
            if 'NODE_NAME' in os.environ:
                name = os.environ['NODE_NAME']
            else:
                name = get_local_ip()
        self.name = name
        print('I am ' + self.name)
        retry = KazooRetry(max_tries=-1)
        self.zk = KazooClient(hosts=server_host, connection_retry=retry, timeout=5.0)
        while True:
            try:
                self.zk.start()
                break
            except Exception as e:
                print(e)
        self.zk.ensure_path('/nodes/' + self.name)
        if activate_distributor:
            self.distributor = Distributor(name=name, kafka_bootstrap_servers=distributor_kafka_host,
                                           trigger=lambda x: self.randomize_location())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.jetson is not None:
            self.jetson.close()
        self.zk.close()

    def randomize_location(self):
        self.latitude = random.uniform(39.253085, 39.308899)
        self.longitude = random.uniform(116.190375, 116.215619)
        if self.distributor:
            self.distributor.distribute(self.latitude, self.longitude).get(timeout=10)

    def get_stats(self) -> Optional[dict]:
        # print(self.jetson)
        if self.jetson is None:
            freq = psutil.cpu_freq()
            return {"GR3D": None,
                    "RAM": {"use": psutil.virtual_memory().used / 1024 / 1024,
                            "tot": psutil.virtual_memory().total / 1024 / 1024, "unit": "M"},
                    "CPU": [{'name': 'CPU' + str(i + 1), 'status': 'ON', 'val': percent, 'frq': freq.current,
                             'governor': 'schedutil'} for i, percent in enumerate(psutil.cpu_percent(percpu=True))]
                    }
        return self.jetson.stats

    @staticmethod
    def get_is_ready(timeout: int = 1) -> bool:
        # try:
        #     requests.get('http://localhost:8081', timeout=timeout)  # TODO
        # except Exception:
        #     return False
        return True

    # def get_services(self):
    #     container_list = docker_client.containers.list()

    @staticmethod
    def get_hard_disk_usage() -> dict:
        usage = shutil.disk_usage('/')
        return {
            'tot': usage.total / 1024 / 1024,
            'use': usage.used / 1024 / 1024,
            'unit': 'MB'
        }

    def get_bandwidth(self) -> dict:
        net_if_stats = psutil.net_if_stats()
        max_bandwidth = self.max_bandwidth
        net_name = None
        for key in net_if_stats:
            stat = net_if_stats[key]
            if stat.isup and key != 'lo':
                net_name = key
                # max_bandwidth = stat.speed # TODO some only reports zero
                break
        if net_name is None:
            return {
                'use': 0,
                'tot': 0,
                'unit': 'Mbps'
            }
        counters = psutil.net_io_counters(pernic=True)[net_name]
        new_bytes = counters.bytes_sent + counters.bytes_recv
        new_timestamp = time.time()
        if self.last_bytes:
            # print(new_bytes - self.last_bytes)
            use = (new_bytes - self.last_bytes) / (new_timestamp - self.last_bytes_timestamp) / 1024. / 1024. * 8
            result = {
                'use': min(use, max_bandwidth),
                'tot': max_bandwidth,
                'unit': 'Mbps'
            }
        else:
            result = {
                'use': 0,
                'tot': max_bandwidth,
                'unit': 'Mbps'
            }
        self.last_bytes = new_bytes
        self.last_bytes_timestamp = new_timestamp
        return result

    def set_coordinate(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def set_max_bandwidth(self, max_bandwidth):
        self.max_bandwidth = max_bandwidth

    def keep_alive(self):
        stats = self.get_stats()
        data = {
            'update_time': datetime.datetime.now().isoformat(),
            'stats': stats,
            'hard_disk': self.get_hard_disk_usage(),
            'bandwidth': self.get_bandwidth(),
            'ip': get_local_ip(),
            'name': self.name,
            'ready': self.get_is_ready(),
            'virtual': self.is_virtual,
            'latitude': float("%0.6f"%self.latitude),
            'longitude': float("%0.6f"%self.longitude),
            'port': 8889,
            'latency': self.latency
            # 'services': self.get_services()
        }
        print(data)
        self.zk.set('/nodes/' + self.name, json.dumps(data).encode())
