from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer


authorizer = DummyAuthorizer()

authorizer.add_user("huchuanwen", "qweasdzxc", "/home/huchuanwen/bishe/collaborativeDL/models", perm="elr")  # adfmw

authorizer.add_anonymous("/home/huchuanwen/bishe/checks")

handler = FTPHandler
handler.authorizer = authorizer

server = FTPServer(("127.0.0.1", 2121), handler)
server.serve_forever()