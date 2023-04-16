---
title: Zombie Processes
date: 2013-06-08 00:00
tags:
  - linux
---

Zombie processes are "dead" processes. That is, processes that have finished executing and are waiting for the parent to reap them (collect information about their status).

You cannot kill a zombie process because they are already dead. To get rid of a zombie process, kill its parent.

Zombie processes are, in some ways, the opposite of [Orphan Processes](orphan-processes.md). When a child process finishes running, its state (PID and return code) sits in the process table, waiting for the parent to collect it by calling `wait(2)`. In this state, we say a child is a *defunct* or *zombie* process.

Therefore, zombie processes have completed execution and thus take up almost no system resources.

However, they do hold on to PIDs that the OS could allocate to other processes. With too many zombie processes - say if a poorly coded program isn't collecting return status information fast enough, it's possible for the OS to run out of PIDs. You can increase the number of available PIDs by modifying the kernel parameter `kernel.pid_max` using either the `sysctl` command or editing the `/etc/sysctl.conf` file. More info is available [here](http://www.cyberciti.biz/tips/howto-linux-increase-pid-limits.html).

To see this in action, I'm going to create a program that makes a child process that runs for 1 second and a parent process that'll sleep indefinitely.

```python
from os import fork, getppid, wait
from sys import exit
from time import sleep

pid = fork()

if pid == 0:
    exit("Child: Goodbye, cruel world")
else:
    print("Parent: I created a child with pid", pid,
          "and now all I want to do is sleep...")
    while True:
        sleep(1)
```

```bash
> python fork_zombie.py
Parent: I created a child with pid 26556, and now all I want to do is sleep.
Child: Goodbye, cruel world!
```

Now, in a separate terminal instance (or a separate Screen, Tmux or
Byobu window) let's examine the child's process using `ps`.

```bash
> ps -ef | grep 26556
lex      26556 26555  0 21:40 pts/3    00:00:00 [python] <defunct>
```

There we have it, a defunct/zombie process that utilizes no memory
awaiting our parent to acknowledge it by calling `wait(2)` or for our
parent to die.

---

You can't really "kill" zombie processes because, well, they're already dead.

For some processes, sending the `SIGCHLD` signal to the parent process
could instruct it to call `wait` and *reap* its dead child processes.
However, we're out of luck if we haven't implemented a handler for `SIGCHLD`.

The only way to clear out a zombie process is for the parent to *reap*
the process by calling `kill` or for the parent process itself to die.
If, however, there's a zombie process in the system for a parent
process that has already stopped, this could be a sign of an
operating system bug.

In our example, we can get rid of the zombie process by closing the
parent process, which I will do by pressing Ctrl + C in the terminal
where my parent is running.

```bash
^CTraceback (most recent call last):
  File "fork.py", line 11, in 
    time.sleep(1)
KeyboardInterrupt
```

Now, if we look for the initial zombie process, it should be nowhere to
be found.

```bash
> ps -e | grep 27439
>
```
