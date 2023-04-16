---
title: Orphan Processes
date: 2013-06-08 00:00
tags:
  - Unix
---

When a parent process finishes executing before its children, the child
processes are said to become orphan processes.

When this happens, the init process - the first process executed by the kernel at boot, with a PID of 1 - adopts the child. The init process then waits for the child process to finish.

In this code example, I have a script that calls [fork System Call](fork-system-call.md) to create a child process. Then, we sleep for a second 3 times after printing its parent's PID in the child process. The parent will exit on the first iteration, thus orphaning its child.

```python
from os import fork, getppid
from time import sleep
 
pid = fork()
 
if pid == 0:
    print("Child: I'm about to become an orphan!")
    for _ in range(3):
        sleep(1)
        print("Child: My parent is", getppid())
else:
    print("Parent: I just created child", pid)
    sleep(1)
```

```bash
> python fork_orphan.py
Parent: I just created child 19683
Child: I'm about to become an orphan!
Child: My parent is 19682
Child: My parent is 1
Child: My parent is 1
```

So, to prevent the child from becoming an orphan, our parent process
can call the [wait System Call](wait System Call).

`wait(2)` effectively waits for its children to exit and then collects information about them. The system call is available in Python as `os.wait()`, which returns a tuple containing the child's PID and exit status indication.

```python
from os import fork, getppid, wait
from sys import exit
from time import sleep

pid = fork()
 
if pid == 0:
    print("Child: Hope my parent doesn't forget me this time!")
    for _ in range(3):
        sleep(1)
        print("Child: My parent is", getppid())
else:
    wait()
    print("Parent: My child has finished processing. My work here is done.")
```

```bash
> python fork_wait.py
Child: Hope my parent doesn't forget me this time!
Child: My parent is 20037
Child: My parent is 20037
Child: My parent is 20037
Parent: My child has finished processing. My work here is done.
```

The operating system uses the procedure of orphaning a process to daemonizes a process.

[@wikipediaOrphanProcess]
