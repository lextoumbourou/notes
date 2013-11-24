Title: Packet capturing in parallel with Ansible
Tagline: And why Ansible continues to blow my mind
Slug: packet-capturing-in-parallel-ansible
Date: 2013-08-13

<a name="intro"></a>

<div class="intro">
Ansible continues to amaze me. It seems to be able to solve so many
problems with such simplicity and I keep finding new uses for it. Enter
today's problem and solution.
</div>

### The Problem

-   Run Tcpdump for a short window in parallel across a massive test
    environment while front-end tests are performed
-   Gzip and tar each generated pcap and copy to a remote share for
    distribution

After getting my environment setup right, the solution took me a total
of 30 minutes to write and deploy. Plus, it is completely reusable in
any environment (though the interface parameter passed to tcpdump may
require tweaking).

### The Solution

1\. First I utilised Ansible's host variables to create a unique file
name for each server.

<script src="https://gist.github.com/lextoumbourou/7623119.js"></script>

2\. Then I kicked off a Tcpdump on each server in parallel. (Ansible
runs 5 parallel processes by default, this can be increased by passing
the `--forks=10` parameter to the `ansible-playbook` script.

<script src="https://gist.github.com/lextoumbourou/7623122.js"></script>

As well as running Playbooks against servers in parallel, tasks can be
run asynchronously by setting the `async` (maximum runtime) and `poll`
(how often to poll the job for status). In this example, I'm using
Ansible's [fire-and-forget][] pattern to run the command without
pausing, allowing me to kill it later on in the Playbook.

3\. Next the script is [paused][] for a minute allowing the testers to
run through their failing tests.

<script src="https://gist.github.com/lextoumbourou/7623129.js"></script>

4\. Then I kill off the tcpdump command with pkill.

<script src="https://gist.github.com/lextoumbourou/7623137.js"></script>

5\. Lastly, I can compress the logs and copy them to my localhost using
the [fetch][] module. The `flat` parameter copies up each file
individually. Without it, the files are stored in a
`${server_name}/{$path}/file` directory structure, ensuring files with
the same name don't overwrite each other.

<script src="https://gist.github.com/lextoumbourou/7623141.js"></script>

It's probably a good idea to clean up the files on the remote server
too.

<script src="https://gist.github.com/lextoumbourou/7623146.js"></script>

6\. Now I run it from the command line...

<script src="https://gist.github.com/lextoumbourou/7623149.js"></script>

And like magic:

<script src="https://gist.github.com/lextoumbourou/7623153.js"></script>

Here's the entire Playbook:

<script src="https://gist.github.com/lextoumbourou/7611499.js"></script>

  [fire-and-forget]: http://www.ansibleworks.com/docs/playbooks2.html#id19
  [paused]: http://www.ansibleworks.com/docs/modules.html#pause
  [fetch]: http://www.ansibleworks.com/docs/modules.html#fetch
  [And we out]: https://twitter.com/lexandstuff
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
