from kazoo.client import KazooClient
import json
from conf.getConf import *
from log import KafkaLog

log = KafkaLog()

def getColDevice():
    class server:
        def __init__(self):
            self.host = ''
            self.port = 0
            self.RAM = 0

        def __str__(self) -> str:
            return self.host + ' ' + str(self.port) + ' ' + str(self.RAM)

    zk = KazooClient(hosts="10.4.10.254:2181")

    zk.start()    #与zookeeper连接
    #makepath=True是递归创建,如果不加上中间那一段，就是建立一个空的节点


    devices = zk.get_children('/nodes')

    server = server()
    for device in devices:
        tmpDeviceInfo = zk.get('/nodes/' + device)[0].decode(encoding='utf-8')
        device = json.loads(tmpDeviceInfo)
        print(device)
        ram = device['stats']['RAM']['tot']-device['stats']['RAM']['use']

        if ram > server.RAM:
            server.host = device['ip']
            server.port = device['port']
            server.RAM = ram


    print(devices)
    setColServer(server.host, server.port)
    log.logSend("INFO " + getLocalhost() + " select device " + server.host)
    zk.stop()

getColDevice()