from twisted.internet import reactor


class CounterManager(object):
    counters = []

    @classmethod
    def add_counter(cls, counter):
        cls.counters.append(counter)

    @classmethod
    def has_active_counters(cls):
        return all([not c.is_active for c in cls.counters])


class Counter(object):
    def __init__(self, name, between_time, counter=5):
        self.name = name
        self.between_time = between_time
        self.counter = counter
        self.is_active = True

        CounterManager.add_counter(self)

    def count(self):
        if self.counter == 0:
            self.is_active = False
            if CounterManager.has_active_counters():
                print 'No counters active. Stopping!'
                reactor.stop()
        else:
            print self.name + ':',  self.counter
            self.counter -= 1
            reactor.callLater(self.between_time, self.count)

c1 = Counter('1', 0.5)
c2 = Counter('2', 1)
c3 = Counter('3', 0.1)


reactor.callWhenRunning(c1.count)
reactor.callWhenRunning(c2.count)
reactor.callWhenRunning(c3.count)

print 'Start'
reactor.run()
