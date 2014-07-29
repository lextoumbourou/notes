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

```while True:
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
