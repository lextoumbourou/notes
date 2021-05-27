Title: What Are Zombie Processes?
Tagline: Unix 101 again because I failed it the first time
Slug: what-are-zombie-processes
Category: Unix
Date: 2013-06-08

<a name="intro"></a>

<div class="intro">
A while back I went for a job interview at a high profile company you've
heard of. Since it was for a system engineer-type role, there were
questions about Unix internals. One of those questions was about zombie
processes. At the time, I hadn't grasped this concept completely and I
recall giving a vague answer about processes whose parent process had
died or something. I didn't get the job.

<p>

In this short article, for my sake if nothing else, I'm going to
clear up the misconceptions I had about zombie/defunct processes,
through a series of explanations and code examples. If you have
experience coding in Python, it'll be helpful, but all the code should
be simple enough that it can be transferred to the scripting language
of your choice.
</div>

* [Intro](#intro)
* [Multiprocessing Fundamentals](#fundamentals)
* [What Zombie Processes Are Not (Spoiler: Orphan Processes)](#what-zombie-processes-are-not)
* [So Then, What Zombie Processes Are](#what-are-zombie-processes)
* [How To "Kill" Zombie Processes](#killing-zombie-processes) 
* [Summary (tl;dr)](#summary)

* * * * *

<a name="fundamentals"></a>

### [Multiprocessing Fundamentals](#fundamentals)

When a process spawns a separate process to handle some work, it invokes
the `fork(2)` system call. `fork(2)` duplicates the current process in
memory and both processes continue executing from the same location. We refer to
the new process as the *child* process. We refer to the calling process
as the *parent*. Type `man 2 fork` for information.

In the parent process, the `fork(2)` call returns the process id, aka
the *PID*, of the child process. The same call from the child returns 0.

Python provides a wrapper around the call called `fork()` from the `os`
module. Let's see it in action.

##### fork.py

    :::python
    from os import fork
    from time import sleep
     
    # Fork the process
    pid = fork()
     
    if pid == 0: # We're in the child process
        print "Child: Waddup?"
    else: # We're in the parent process
        print "Parent: I just created child", pid
     
        # Sleep for a second to avoid being dropped back to the shell when the parent finishes
        sleep(1)

Now, when I run the script in the terminal, we should see the child
process' PID followed immediately by the code executed in the
child process.

    :::bash
    > python fork.py
    Parent: I just created child 19478
    Child: Waddup?

From the child process, we can get the parent process' PID by calling
the `getppid(2)` system call. Which, again, Python provides a wrapper
around called `os.getppid()`.

##### fork_ppid.py

    :::python
    from os import fork, getppid
    from time import sleep
     
    pid = fork()
     
    if pid == 0: # We're in the child process
        print "Child: Waddup?"
        print "Child: My parent is", getppid()
    else:
        print "Parent: I just created child", pid
        # Sleep for a second to avoid being dropped back to the shell when the parent finishes
        sleep(1)
</p>

    :::bash
    > python fork_ppid.py
    Parent: I just created child 19741
    Child: Waddup?
    Child: My parent is 19740

Of course, we are not limited to just one child process; we can have as
many as we like.

* * * * *

<a name="what-zombie-processes-are-not"></a>

### [What Zombie Processes Are Not (Spoiler: Orphan Processes)](#what-zombie-processes-are-not)

When a parent process finishes executing before its children, the child
processes are said to become *orphan* processes. When this happens, the
child is *adopted* by the *init* process - the first process executed by
the kernel at boot - which has a PID of 1. The *init* process then waits
for the child process to finish.

Let me add a line to execute in the child process that sleeps for a
second 3 times after printing its parent's PID. The parent will exit on the first iteration, thus *orphaning* its child.

##### fork_orphan.py

    :::python
    from os import fork, getppid
    from time import sleep
     
    pid = fork()
     
    if pid == 0:
        print "Child: I'm about to become an orphan!"
        for _ in range(3):
            sleep(1)
            print "Child: My parent is", getppid()
    else:
        print "Parent: I just created child", pid
        sleep(1)

Let's take a look at the results.

    :::bash
    > python fork_orphan.py
    Parent: I just created child 19683
    Child: I'm about to become an orphan!
    Child: My parent is 19682
    Child: My parent is 1
    Child: My parent is 1

So, to prevent the child from becoming an `orphan`, our parent process
can call the `wait(2)`  system call. `wait(2)` effectively waits
for its children to exit then collects some information about them. In
Python, the system call is available as `os.wait()` which returns a
tuple containing the child's PID and exit status indication (see docs
for more info).

##### fork_wait.py

    :::python
    from os import fork, getppid, wait
    from sys import exit
    from time import sleep
     
    pid = fork()
     
    if pid == 0:
        print "Child: Hope my parent doesn't forget me this time!"
        for _ in range(3):
            sleep(1)
            print "Child: My parent is", getppid()
    else:
        wait()
        print "Parent: My child has finished processing. My work here is done."


And when we run it?

    :::bash
    > python fork_wait.py
    Child: Hope my parent doesn't forget me this time!
    Child: My parent is 20037
    Child: My parent is 20037
    Child: My parent is 20037
    Parent: My child has finished processing. My work here is done.

The procedure of orphaning a process is used by the operating system to daemonises a process (which is a topic for another day).

* * * * *

<a name="what-are-zombie-processes"></a>

### [So Then, What Zombie Processes Are](#what-are-zombie-processes)

Zombie processes are, in some ways, the opposite of orphaned processes.
When a child process finishes running, it's state (PID and return code)
sit in the process table waiting for the parent to collect it by
calling `wait(2)`. In this state, a child is said to become - wait for it - a *defunct* or
*zombie* process. Therefore, <span class="pull_quote right">zombie processes have completed execution 
and thus take up almost no system resources</span>. However, they do hold on to
PIDs that could potentially be allocated to other processes and, with too
many zombie processes - say if a poorly coded program isn't collecting
return status information fast enough, it's possible for the OS to run out of
PIDs. You can increase the number of available PIDs by modifying the
kernel parameter `kernel.pid_max` using either the `sysctl` command or by editing the `/etc/sysctl.conf` file. More info
is available [here](http://www.cyberciti.biz/tips/howto-linux-increase-pid-limits.html).

To see this in action, I'm going to create a program which creates a child process that runs
for 1 second and a parent process that'll sleep indefinitely.

##### fork_zombie.py

    :::python
    from os import fork, getppid, wait
    from sys import exit
    from time import sleep
     
    pid = fork()
     
    if pid == 0:
        exit("Child: Goodbye, cruel world")
    else:
        print "Parent: I created a child with pid", pid,\
              "and now all I want to do is sleep..."
        while True:
            sleep(1)

</p>

    :::bash
    > python fork_zombie.py
    Parent: I created a child with pid 26556 and now all I want to do is sleep...
    Child: Goodbye cruel world!

Now, in a separate terminal instance (or a separate Screen, Tmux or
Byobu window) let's examine the child process using `ps`.

    :::bash
    > ps -ef | grep 26556
    lex      26556 26555  0 21:40 pts/3    00:00:00 [python] <defunct>

There we have it, a defunct/zombie process that utilises no memory
awaiting our parent to acknowledge it by calling `wait(2)` or for our
parent to die.

* * * * *

<a name="killing-zombie-processes"></a>

### [How To "Kill" Zombie Processes](#killing-zombie-processes)

You can't really "kill" zombie processes because, well, they're already dead.

For some processes, sending the `SIGCHLD` signal to the parent process
could instruct it to call `wait` and *reap* its dead child processes.
However, if a handler hasn't been implemented for `SIGCHLD`, then you're
outta luck.

The only way to clear out a zombie process is for the parent to *reap*
the process by calling `kill` or for the parent process itself to die.
If, however, there's a zombie processes in the system for a parent
process that's already finished executing, this could be a sign of an
operating system bug.

In our example, we can get rid of the zombie process by closing the
parent process, which I will do by pressing Ctrl + C in the terminal
where my parent is running.

    :::bash
    ^CTraceback (most recent call last):
      File "fork.py", line 11, in 
        time.sleep(1)
    KeyboardInterrupt

Now, if we look for the initial zombie process, it should be no where to
be found.

    :::bash
    > ps -e | grep 27439
    >

* * * * *

<a name="summary"></a>

### [Summary (tl;dr)](#summary)

To recap, zombie processes are not orphan processes, they are dead processes: processes that have finished executing and are waiting for the parent to reap them (collect information about their status). You are barking up the wrong tree trying to kill a zombie process because they are already dead. To get rid of a zombie process, kill its parent. Hope that helps.
