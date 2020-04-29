#pip install python-consul
import consul
import time

class Consul(object):
    def __init__(self, host, port):
        '''初始化，连接consul服务器'''
        self._consul = consul.Consul(host, port)

    def register(self, server_name, ip, port, tags):
        c = consul.Consul()  # 连接consul 服务器，默认是127.0.0.1，可用host参数指定host
        print(f"开始注册服务{server_name}")
        check = consul.Check.tcp(ip, port, "10s")  # 健康检查的ip，端口，检查时间
        c.agent.service.register(server_name, f"{server_name}-{ip}-{port}",
                                 address=ip, port=port, tags=tags
                                 # check = check,
                                 )  # 注册服务部分
        print(f"注册服务{server_name}成功")

    def RegisterService(self, name, host, port, tags=None):
        tags = tags or []
        # 注册服务

        self._consul.agent.service.register(
            name,
            name,
            host,
            port,
            tags,
            # 健康检查ip端口，检查时间：5,超时时间：30，注销时间：30s
            # check=consul.Check().tcp(host, port, "5s", "30s", "30s")
        )

    def GetService(self, name):
        services = self._consul.agent.services()
        service = services.get(name)
        if not service:
            return None, None
        addr = "{0}:{1}".format(service['Address'], service['Port'])
        return service, addr



if __name__ == '__main__':
    host="127.0.0.1" #consul服务器的ip
    port="8500" #consul服务器对外的端口
    consul_client=Consul(host,port)

    name="DLserver"
    port = 8501
    tags = ["100"]
    consul_client.register(name,host,port,tags)

    # consul_client._consul.kv.put('foo','bar')
    #
    # check = consul.Check().tcp(host, port, "5s", "30s", "30s")
    # print(check)
    # res=consul_client.GetService("maple2")
    # print(res)
    services = consul_client._consul.agent.services()
    # print(services)
    ip = ""
    port = 0
    tag = 0
    for i in services:
        if i.startswith("DLserver"):
            if int(services[i]["Tags"][0]) > tag:
                tag = int(services[i]["Tags"][0])
                ip = services[i]["Address"]
                port = services[i]["Port"]

    print("ip {0} port {1} tag{2}".format(ip,port,tag))