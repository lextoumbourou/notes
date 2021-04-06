from twisted.internet import reactor
from twisted.internet.protocol import Protocol, ClientFactory

import sys


class PostOfficeProtocol(Protocol):
    def dataReceived(self, data):
        sys.stdout.write(data)

class PostOfficeFactory(ClientFactory):
    protocol = PostOfficeProtocol


reactor.connectTCP("localhost", 10000, PostOfficeFactory())
reactor.run()
