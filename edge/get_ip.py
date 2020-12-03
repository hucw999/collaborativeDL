import socket
import struct
import sys
import time
import traceback
from subprocess import Popen
from threading import Thread
import os

HEADER = b'\xEF\xAE'
PORT = 14444
INTERVAL = 2

sysname = os.uname().sysname


def linux_only(func):
    def wrapper(*args, **kwargs):
        if sysname != 'Linux':
            return
        return func(*args, **kwargs)

    return wrapper


@linux_only
def create_chain():
    Popen("iptables -t nat -N forward-port-to-master".split()).wait()
    if Popen('iptables -t nat -C OUTPUT -j forward-port-to-master'.split()).wait():
        Popen("iptables -t nat -I OUTPUT -j forward-port-to-master".split()).wait()


@linux_only
def flush():
    Popen("iptables -t nat -F forward-port-to-master".split()).wait()


@linux_only
def add_rule(local_ip, remote_ip, port):
    Popen(
        f"iptables -t nat -A forward-port-to-master --dst {local_ip} -p tcp --dport 2{port} -j DNAT --to-destination {remote_ip}:{port}".split()).wait()


class GetIp:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.connect(('40.253.65.211', 53))
        self.local_ip = self.server.getsockname()[0]
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind(('0.0.0.0', PORT))
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.node_id = struct.unpack("!I", socket.inet_aton(self.local_ip))[0]
        self.master_ip = self.local_ip
        self.last_recv_time = 0
        create_chain()
        self.update()

    def listener(self):
        while True:
            data, peer_addr = self.server.recvfrom(6)
            if len(data) != 6 or data[:2] != HEADER or peer_addr[1] != PORT:
                continue
            peer_id = struct.unpack('!I', data[2:])[0]
            if peer_id > self.node_id:
                print(f'网关节点ip:{peer_addr[0]}')
                if self.master_ip != peer_addr[0]:
                    self.master_ip = peer_addr[0]
                    self.update()
                self.last_recv_time = time.time()
            # elif peer_id == self.node_id:
            #     print(f"Error node id conflict: {peer_addr[0]} node ID:{peer_id}", file=stderr)
            elif peer_id < self.node_id:
                print('recv peer id:', peer_id)

    def sender(self):
        while True:
            try:
                if self.master_ip == self.local_ip or time.time() - self.last_recv_time >= 2.5 * INTERVAL:
                    if self.master_ip != self.local_ip:
                        self.master_ip = self.local_ip
                        self.update()
                    data = HEADER + struct.pack('!I', self.node_id)
                    self.server.sendto(data, ('255.255.255.255', PORT))
                    print(f"{self.master_ip}: 当选网关节点")
            except Exception as e:
                print(e)
                traceback.print_exc()
            time.sleep(INTERVAL)

    def update(self):
        flush()
        add_rule(self.local_ip, self.master_ip, 9092)
        add_rule(self.local_ip, self.master_ip, 2181)
        add_rule(self.local_ip, self.master_ip, 8500)


if __name__ == '__main__':
    ip = GetIp()
    Thread(target=ip.listener, daemon=True).start()
    ip.sender()
