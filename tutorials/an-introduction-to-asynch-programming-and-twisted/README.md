# An Introduction To Asynchronous Programming and Twisted

## Part 1

* Synchronous model
	<img src="./images/synchronous-model.png"></img>
* Threaded model:
	<img src="./images/threaded-model.png"></img>
* Async model
	<img src="./images/async-model.png"></img>

* Condition under which asynchronous model is faster - when particular task is waiting for a resource (ie blocking).
* Fundamental idea behind asynchronous model: "when faced with a task that would normally block...(it) will instead execute some other task that can still make progress"
* Async works best when:
    * Large number of tasks, so one should always be making progress
    * Tasks do heaps of I/O, causing sync program to block when it could otherwise be serving requests
    * Tasks are largely independant from each other (so little need for inter-task communication)

## Part 2

* Blocking client basically just waits until socket no 1 has finished sending data, then goes to no 2 and so on.
* Async client moves to the next socket each time there's a block, using:

```
try:
		new_data = sock.recv(1024)
except socket.error, e:
		if e.args[0] == errno.EWOULDBLOCK:
				# this error code means we would have
				# blocked if the socket was blocking.
				# instead we skip to the next socket
				break # Move to next socket
```

* Contrast with Go's model:
    * Non-blocking isn't really needed. It has "multiple threads of execution" so you can leave a thread blocked and do work in another
* The use of a loop that waits for events to happen then acts on them called a "Reactor Pattern".
<img src="./images/reactor-pattern.png"></img>
    * Called "reactor" because it waits for and reacts to events
    * Twisted is basically an implementation of the Reactor Pattern with extras.

## Part 3

* Reactor loop doesn't start until told to. You start by calling ```reactor.run()```
* Reactor loop runs in same thread as started in.
* Once loop starts, it doesn't stop until killed
* Reactor isn't created - just imported
    * Reactor is a singleton
* Twisted contains multiple reactors. You need to install them before using them.
```
from twisted.reactor import pollreactor
pollreactor.install() # Need to 'install' reactors before using them

from twisted.internet import reactor
reactor.run()
```
* Properties of callbacks:
    * Callback code runs in same thread as Twisted loop
    * When callbacks are running, Twisted loop isn't
    * Vice versa
    * Reactor loop resumes when callbacks are done 
* Avoid making block calls in call backs.
    * Network I/O blocking is handled by Twisted out the box
    * Use Twisted API for stuff like ```os.system``` which is non-blocking
* Example of callLater(<call_time_in_seconds>, <method>)
```
class Countdown(object):
    counter = 5

    def count(self):
        if self.counter == 0:
            reactor.stop()
        else:
            print self.counter, '....'
            self.counter -= 1
            reactor.callLater(0.1, self.count)

reactor.callWhenRunning(Countdown().count)

print 'Start'
reactor.run()
print 'Stop!'
```
* Exceptions raised in a call back won't cause the app to die. It'll report it and continue.

### Exercises

* [3 independent counters example](./exercises/part_3_ex_1.py)
* [LoopingClass example](./exercises/part_3_ex_2.py)

## Part 4

* ```reactor.addReader(self)``` takes an object that much conform to an [interface](http://twistedmatrix.com/trac/browser/tags/releases/twisted-8.2.0/twisted/internet/interfaces.py).
* The ```addReader``` method should take an an ```IReadDescriptor``` provider

### Exercises

* The first exercise basically just requires the prevention of the task calling ```reactor.addReader(self)``` when the socket connection fails eg: ```try / except socket.error```
* [Use callLater to make the client timeout](./exercises/part_4_ex_2.py) 

## Part 5

* Twisted architecture is composed of layers of abstraction
	* Example: IReadDescriptor - "file descriptor you can read bytes from"
* Transports
	* Defined by ```ITransport``` in main ```interfaces``` module
	* A Transport is a single connection that can send or receive bytes (TCP, UDP, Unix pipes)
	* Generally use implementations that Twisted provide
* Protocols
	* Implements a networking protocol like FTP or IMAP (or something you create)
* Protocol Factories
	* Creates instances of Protocol Factories
	* ```buildProtocol``` method should return a new Protocol instance when it's called
	* Primary Factory class is ```twisted.internet.protocol.Factory``` but will generally use a more specialised class
* Protocol construction
	* Step 1
		* In Protocol Factory, need to specify Protocol to build with the ```protocol``` class var.
		* Using ```buildProtocol``` method, we instantiat the protocol as above.
		* Protocol created with same factory can share state with parent factory
	* Step 2
		* Protocol connects with a Transport using the ```makeConnection``` method. Stores a ref to the transport object in the ```self.transport``` attribute
	* Step 3
		* Protocol can process incoming data with ```dataReceived```, which takes as input a sequence of bytes (aka a string)
		* ```self.transport.getPeer()``` returns information on the server data is coming from

### Exercises

* [get poetry client with timeout](./exercises/part_5_ex_1.py)

## Part 6

* Can't mix synchronous code with async (obvious)
* Use ```Failure``` object for handling passing errors back instead of callbacks (even better -- another abstraction discussed in later Parts)

## Part 7

* Consider that if errors aren't caught in async programs, program can become "stuck" waiting for callback that may never arrive
* ```Deferred``` class is designed to make dealing with callbacks easier
  * Defined in ```twisted.internet.defer```
  * Contains a chain of callbacks and errbacks
  * Will not let you fire the normal result callback twice
    ```
    d.addCallbacks(out, out)
    d.callback('First result')
    d.callback('Second result') # This will error with twisted.internet.defer.AlreadyCalledError
    ```
  * ```addBoth``` method is used to add same function to callback and errback handlers

## Part 8

* More about deferreds, contrasting synchronous programming returning actual results, with deferred returning something that'll fire later. Nothing new.

### Exercises

* [Exercise 1: Timeout if poem isn't received in time](./exercises/part_8_ex_1.py)
* [Exercise 2: Print appropriate address on failure](./exercises/part_8_ex_2.py)

## Part 9

* low-context code - code like `socket.connect()` that is for general purposes and knows nothing about the functions from which it came form
* high-context code - code like `main()` which knows all about the program and its purpose
* Deferred class has tools to help catch general errors that wasn't explicitely caught and send to `errBack`. When an exception is caught, it will automatically fire the next `errBack` in the deferred.
* Callbacks and errBacks are always registered in pairs:
   * `addCallbacks` - takes in two arguments a callBack and an errBack
   * `addCallback` - takes in 1 argument (plus arguments to pass into callBack). It adds an implicit pass-through `errback`, which just falls-through to next errback.
   * `addErrback` - same but with implicit `callBack`
   * `addBoth` - takes in 1 argument that's both a callback and errback

## Part 10: Poetry Transformed

* Callback chains
  * ```d.addCallback(do_something_that_could_error_or_succeed)``` - if it asserts, it will pass to next errback. Else, pass result to callback
  * ```d.addCallbacks(handle_success, handle_err)``` 
* On errBack, returning a value will be passed to the next callBack in the chain, and thus continue the callBack/errBack chain.  

## Part 11: Your Poetry is Served

* ```self.transport.write()``` - write data to connection.
* ```self.transport.loseConnection()``` - Close connections after writing all data.
* ```reactor.listenTCP``` method takes in an subclass instance of ```ServerFactory```,
* ```Factory``` -- builds a protocol and saves persistant configuration. Does not need to know about network at all.
* ```Protocol``` -- actually implements protocol handling.
* ```reactor.listenTCP(port, factory)``` - method for ensuring a factory listens on the wire.
  * Creates a "listening socket and adds it to the event loop"
  * Twisted will create a ```Transport``` via the ```Factory``` class, which transports data to the ```Protocol```

## Part 12: A Poetry Transformation Server

* Separate protocol logic from service logic is a common pattern.
  * Allows for providing the same service via multiple protocols.
  * Question: under what circumstance would you use the same service via multiple protocols?
* Smart to guard access to service through `xform` method prefix to prevent attackers from running arbitrary methods.

### Exercises

1. If client sends a malformed netstring, q ``NetstringParseError`` is raised, causing the ``self.transport.loseConnection()`` method to be called, dropping the connection.
<<<<<<< HEAD
2. [Halve method](./exercises/part_12_ex_2.py#L92) (test: ``echo -n "21:halve.HERE IS MY POEM,"  | nc localhost $1``)
3. [Second protocol](./exercises/part_12_ex_3.py)
