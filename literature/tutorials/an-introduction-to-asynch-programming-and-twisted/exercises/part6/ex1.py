from twisted.python.failure import Failure
from twisted.internet import reactor
from twisted.internet.protocol import Protocol, ClientFactory


class TimeoutError(Exception):
    pass


class PostOfficeProtocol(Protocol):

    poem = ''

    def __init__(self, factory):
        from twisted.internet import reactor
        self.factory = factory
        self.timeout = reactor.callLater(2, self.handle_timeout)

    def dataReceived(self, data):
        self.poem += data

    def connectionLost(self, reason):
        if self.timeout.active():
            self.timeout.cancel()
            self.poemReceived(self.poem)

    def poemReceived(self, poem):
        self.factory.poem_finished(poem)

    def handle_timeout(self):
        self.factory.errback(Failure(TimeoutError()))
        self.transport.loseConnection()


class PostOfficeClientFactory(ClientFactory):

    def __init__(self, callback, errback):
        self.callback = callback
        self.errback = errback

    def buildProtocol(self, addr):
        return PostOfficeProtocol(self)

    def poem_finished(self, poem):
        self.callback(poem)

    def clientConnectionFailed(self, connector, reason):
        self.errback(reason)


def get_poetry(host, port, callback, err):
    from twisted.internet import reactor

    factory = PostOfficeClientFactory(callback, err)
    reactor.connectTCP(host, port, factory)


def poetry_main():
    addresses = [('localhost', 10004)]

    poems = []

    def handle_error(err):
        trapped = err.trap(TimeoutError)
        if trapped is TimeoutError:
            print "Timed out, yo."
            print error

        print err

    def got_poem(poem):
        poems.append(poem)
        if len(poems) == len(addresses):
            reactor.stop()

    for address in addresses:
        host, port = address
        get_poetry(host, port, got_poem, handle_error)

    reactor.run()

    for poem in poems:
        print poems


poetry_main()
