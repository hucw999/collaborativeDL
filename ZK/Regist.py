from kazoo.client import KazooClient
import json

zk = KazooClient(hosts="10.4.10.239:2181")

zk.start()    #与zookeeper连接

zk.get_children('/nodes')

