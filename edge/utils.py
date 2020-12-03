import socket
from kazoo.client import KazooClient

# import docker
# docker_client = docker.Client()
#
#
# def start_docker():
#     """
#     Start ml service
#     :return:
#     """
#     docker_client.containers.run("tf-nano")

def get_host_ip(gateway: str = '40.253.65.211', port: int = 80) -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((gateway, port))
        ip = s.getsockname()[0]
    finally:
        s.close()
    # return ip
    return '10.4.10.254'

def get_local_ip(gateway: str = '40.253.65.211', port: int = 80) -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((gateway, port))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def deleteNode():

    zk = KazooClient(hosts='10.4.10.239')
    zk.start()
    zk.delete('/nodes/10.4.10.213')
    zk.stop()
# print(get_host_ip())
# deleteNode()