import configparser
import os
conf = configparser.ConfigParser()
confFile = os.path.dirname(__file__)+'/inf.conf'
conf.read(confFile)

def setColServer(host,port):
    conf.set("inf.server", "host", host)
    conf.set("inf.server", "port", str(port))
    conf.write(open(confFile,'w'))

def getKafkaHosts():

    return conf.get("kafka.server", "hosts")

def getConsoleTopic():
    return conf.get("kafka.server", "consoleChanel")

def getLocalhost():
    return conf.get("inf.local", "host")

def getColServer():
    host = conf.get("inf.server", "host")
    port = conf.getint("inf.server", "port")
    return host, port

def getZkHosts():
    return conf.get("zk.server", "hosts")