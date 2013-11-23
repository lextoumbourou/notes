Title: What Are Zombie Processes?
Slug: what-are-zombie-processes
date: 2013-06-08

[Intro][]  

[Multiprocessing Fundamentals][]  

[What Zombie Processes Are Not (Spoiler: Orphan Processes)][]  

[So Then, What Zombie Processes Are][]  

["Killing" Zombie Processes][]  

[Summary][]  

</p>

* * * * *

</p>

<a name="intro"></a>

### Intro

</p>

A while back I went for a job interview at a high profile company you've
heard of. Since it was for a sys engineer-type role, there were
questions about Unix internals. One of those questions was about zombie
processes. At the time, I hadn't grasped this concept completely and I
recall giving a vague answer about processes whose parent process had
died or something. I didn't get the job.

</p>

In this short article, for my sake if for nothing else, I'm going to
clear up the misconceptions I had about zombie/defunct processes,
through a series of explanations and code examples. If you have
experience coding in Python, it'll be helpful, but all the code should
be simple enough that it can be transferred to your scripting language
of choice.

</p>

* * * * *

</p>

<a name="fundamentals"></a>

### Multiprocessing Fundamentals

</p>

When a process spawns a separate process to handle some work, it invokes
the `fork(2)` system call. `fork(2)` duplicates the current process in
memory and begins executing it from the current location. We refer to
the new process as the *child* process. We refer to the calling process
as the *parent*. Type `man 2 fork` for information.

</p>

In the parent process, the `fork(2)` call returns the process id, aka
the *PID*, of the child process. The same call from the child returns 0.

</p>

Python provides a wrapper around the call called `fork()` from the `os`
module. Let's see it in action.

</p>

<p>
    > cat fork.pyfrom os import forkfrom time import sleep# Fork the processpid = fork()if pid == 0: # We know we're in the child process    print "Child process up in this."else: # We're in the parent process    print "Parent here, I just created child", pid    # Sleep for a second to avoid being dropped back to the shell when the parent finishes    sleep(1)

</p>

Now, when I run the script in the terminal, we should see the child
process' PID followed immediately by the code to be executed in the
child process.

</p>

<p>
    > python fork.pyParent here, I just created child 19478Child process up in this.

</p>
</p>

From the child process, we can get the parent process' PID by calling
the `getppid(2)` system call. Which, again, Python provides a wrapper
around called `os.getppid()`.

</p>

<p>
    > cat fork_ppid.pyfrom os import fork, getppid# ...if pid == 0:    print "Child process up in this."    print "My parent is", getppid()# ...

</p>

<p>
    > python fork_ppid.pyParent here, I just created child 19741Child process up in this.My parent is 19740

</code>

</p>

Of course, we are not limited to just one child process; we can have as
many as we like.

</p>

* * * * *

</p>

<a name="what-zombie-processes-are-not"></a>

### What Zombie Processes Are Not (Spoiler: Orphan Processes)

</p>

When a parent process finishes executing before its children, the child
processes are said to become *orphan* processes. When this happens, the
child is *adopted* by the *init* process - the first process executed by
the kernel at boot - which has a PID of 1. The *init* process then waits
for the child process to finish.

</p>

Let me add a line to execute in the child process that sleeps for a
second 3 times after printing its parent's PID, leaving the parent to
exit; thus *orphaning* its child.

</p>

<p>
    > cat fork_orphan.py#...if pid == 0:    print "I'm about to become an orphan!"    for _ in range(3):        sleep(1)        print "My parent is", getppid()#...

</p>
</p>

Let's take a look at the results.

</p>

<p>
    > python fork_orphan.pyParent here, I just created child 19683Child process up in this.Right now, my parent is 19682> Right now, my parent is 1Right now, my parent is 1

</p>

* * * * *

</p>

So, to prevent the child from becoming an `orphan`, our parent process
can call the `wait(2)` blocking system call, `wait(2)` effectively waits
for its children to exit then collects some information about them. In
Python, the system call is available as `os.wait()` which returns a
tuple containing the child's PID and exit status indication (see docs
for more info).

</p>

<p>
    > cat fork_wait.pyfrom os import fork, getppid, wait# ...pid = fork()if pid == 0:    print "Hope my parent doesn't forget me this time!"    for _ in range(3):        sleep(1)        print "Right now, my parent is", getppid()else:    wait()    print "My child has finished processing. My work here is done."

</p>

And when we run it?

</p>

<p>
    > python fork_wait.pyHope my parent doesn't forget me this time!Right now, my parent is 20037Right now, my parent is 20037Right now, my parent is 20037My child has finished processing. My work here is done.

</p>
</p>

The process of orphaning a process is used by the operating system when
it daemonises a process (which is a topic for another day).

</p>

* * * * *

</p>

<a name="what-are-zombie-processes"></a>

### So Then, What Zombie Processes Are

</p>

Zombie processes are, in some ways, the opposite of orphaned process.
When a child process finishes running, it's state (PID and return code)
sit in the process table waiting for the parent process to collect it by
calling `wait(2)`. In this state, a child is said to be a *defunct* or
*zombie* process. Therefore, zombie processes are "finished" processes
and thus take up almost no system resources. However, they do hold on to
PIDs that could potentially be allocated to other processes and with too
many zombie processes - say if a poorly coded program isn't collect
return status info fast enough - it's possible for the OS to run out of
PIDs. You can increase the number of available PIDs by modifying the
kernel parameter `kernel.pid_max` using either the `sysctl` command or
permanently modify it by editing the `/etc/sysctl.conf` file. More info
is available [here][].

</p>

So, to see this in action, I'm going to create a child process that runs
for 1 second and in the parent process we'll sleep indefinitely.

</p>

<p>
    > cat fork_zombie.pyfrom os import fork, getppid, waitfrom sys import exitfrom time import sleeppid = fork()if pid == 0:    exit("Goodbye, cruel world")else:    print "I created a child:", pid    print "and now all I want to do is sleep..."    while True:        sleep(1)

</p>

<p>
    > python fork.pyI created a child and now all I want to do is sleep…Goodbye cruel world!

</p>

Now, in a separate terminal instance (or a separate Screen, Tmux or
Byobu window) let's examine the child process using `ps`.

<p>
    lex@xbmc-server:/etc/ansible> ps -ef | grep 26556lex      26556 26555  0 21:40 pts/3    00:00:00 [python] <defunct>

</p>

There we have it, a defunct/zombie process that utilises no memory
awaiting our parent to acknowledge it by calling `wait(2)` or for our
parent to die.

</p>

<a name="killing-zombie-processes"></a>

### "Killing" Zombie Processes

</p>

You can't really "kill" zombie processes because there isn't really
anything to kill; they're already dead.

</p>

For some processes, sending the `SIGCHLD` signal to the parent process
could instruct it to call `wait` and *reap* it's dead child processes.
However, if a handler hasn't been implemented for `SIGCHLD`, then you're
outta luck.

</p>

The only way to clear out a zombie process is for the parent to *reap*
the process by calling `kill` or for the parent process itself to die.
If, however, there's a zombie processes in the system for a parent
process that's already finished running, this could be a sign of an
operating system bug.

</p>

In our example, we can get rid of the zombie process by closing the
parent process, which I will do by pressing Ctrl + C in the terminal
where my parent is living.

</p>

<p>
    ^CTraceback (most recent call last):  File "fork.py", line 11, in     time.sleep(1)KeyboardInterrupt

</code>

</p>

Now, if we look for the initial zombie process, it should be no where to
be seen.

</p>

<p>
    > ps -e | grep 27439>

</p>

* * * * *

</p>

<a name="summary"></a>

### Summary (tl;dr)

</p>

To recap, zombie processes are not orphan processes, they are dead
processes: processes that have finished executing and are waiting for
the parent to *reap* them or collect information about their status. You
are barking up the wrong tree trying to kill a zombie process because
they are already dead. To get rid of a zombie process, kill it's parent.
Hope that helps.

</p>

[Love you][].

</p>

<div id="disqus_thread">
</div>
</p>

<p>
<noscript>
Please enable JavaScript to view the [comments powered by Disqus.][]

</noscript>
</p>
[comments powered by <span class="logo-disqus">Disqus</span>][]

  [Intro]: http://feeds.feedburner.com/lextoumbourou#intro
  [Multiprocessing Fundamentals]: http://feeds.feedburner.com/lextoumbourou#fundamentals
  [What Zombie Processes Are Not (Spoiler: Orphan Processes)]: http://feeds.feedburner.com/lextoumbourou#what-zombie-processes-are-not
  [So Then, What Zombie Processes Are]: http://feeds.feedburner.com/lextoumbourou#what-are-zombie-processes
  ["Killing" Zombie Processes]: http://feeds.feedburner.com/lextoumbourou#killing-zombie-processes
  [Summary]: http://feeds.feedburner.com/lextoumbourou#summary
  [here]: http://www.cyberciti.biz/tips/howto-linux-increase-pid-limits.html
  [Love you]: https://twitter.com/lexandstuff
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
