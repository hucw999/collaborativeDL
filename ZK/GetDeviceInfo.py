from kazoo.client import KazooClient
import json

zk = KazooClient(hosts="10.4.10.239:2181")

zk.start()    #与zookeeper连接
#makepath=True是递归创建,如果不加上中间那一段，就是建立一个空的节点

deviceInfo = {'ip':'10.4.10.194', 'port':8502}

deviceInfo = json.dumps(deviceInfo)

# zk.create('/registry/device1', deviceInfo.encode(encoding='utf-8') ,makepath=True)

# zk.create('/registry/device2', deviceInfo.encode(encoding='utf-8') ,makepath=True)

devices = zk.get_children('/registry')

for device in devices:
    tmpDeviceInfo = zk.get('/registry/' + device)[0].decode(encoding='utf-8')
    port = json.loads(tmpDeviceInfo)['port']
    print(port)

# node = zk.get('/registry/device1')[0]  # 查看根节点有多少个子节点
#
# device1 = json.loads(node.decode(encoding='utf-8'))

print(devices)
zk.stop()