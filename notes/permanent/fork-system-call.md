---
title: fork System Call
date: 2013-06-08 00:00
tags:
  - Linux
---

When a process spawns a separate process to handle some work, it invokes
the `fork(2)` system call. `fork(2)` duplicates the current process in
memory, and both processes continue executing from the current location.

We refer to the new process as the *child* process. We refer to the calling process as the *parent*. See `man two fork` for information.

In the parent process, the `fork(2)` call returns the process id, aka
the *PID*, of the child process. The same call from the child returns 0.

Python provides a wrapper around the call called `fork()` from the `os`
module.

In this example, I write a simple script that calls `fork` and then prints whether it's in the child process or the parent.

```python
from os import fork
from time import sleep

# Fork the process
pid = fork()

if pid == 0: # We're in the child process
    print("Child: Waddup?")
else: # We're in the parent process
    print("Parent: I just created child", pid)

    # Sleep for a second to avoid being dropped back to the shell when the parent finishes
    sleep(1)
```

Now, when I run the script in the terminal, we should see the child
process' PID followed immediately by the code executed in the
child process.

```bash
> python fork.py
Parent: I just created child 19478
Child: Waddup?
```

We can get the parent process' PID from the child process by calling
the `getppid(2)` system call. Which Python provides a wrapper
around called `os.getppid()`.

```python
from os import fork, getppid
from time import sleep

pid = fork()

if pid == 0: # We're in the child process
    print("Child: Waddup?")
    print("Child: My parent is", getppid())
else:
    print("Parent: I just created child", pid)
    # Sleep for a second to avoid being dropped back to the shell when the parent finishes
    sleep(1)
```

```bash
> python fork_ppid.py
Parent: I just created child 19741
Child: Waddup?
Child: My parent is 19740
```

Of course, we can have more than one child process; we can have as many as the operating system allows.

[@linuxManPagesFork2]
