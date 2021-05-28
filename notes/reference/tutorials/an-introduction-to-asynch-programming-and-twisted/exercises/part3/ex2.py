from twisted.internet import reactor, task


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

    def start(self):
        self.loop_handler = task.LoopingCall(self.count)
        self.loop_handler.start(self.between_time)

    def count(self):
        if self.counter == 0:
            self.is_active = False
            self.loop_handler.stop()
            if CounterManager.has_active_counters():
                print 'No counters active. Stopping!'
                reactor.stop()
        else:
            print self.name + ':',  self.counter
            self.counter -= 1

print 'Start'

Counter('1', 0.5).start()
Counter('2', 1).start()
Counter('3', 0.1).start()

reactor.run()
