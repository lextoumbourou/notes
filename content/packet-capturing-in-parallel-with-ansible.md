Title: Packet capturing in parallel with Ansible
Slug: packet-capturing-in-parallel-with-ansible
Date: 2012-08-13

Ansible continues to amaze me. It seems to be able to solve so many
problems with such simplicity and I keep finding new uses for it. Enter
today's problem and solution.

### The Problem

</p>

-   Run Tcpdump for a short window in parallel across a massive test
    environment while front-end tests were performed
-   Gzip and tar each generated pcap and copy to a remote share for
    distribution

</p>

After getting my environment setup right, the solution took me a total
of 30 minutes to write and deploy. Plus, it is completely reusable in
any environment (though the interface parameter passed to tcpdump may
require tweaking).

</p>

### The Solution

</p>

\1. First I utilised Ansible's host variables to create a unique file
name for each server.

</p>

<p>
    - hosts: all  sudo: yes  vars:      cap_file: packet_capture_{{ ansible_hostname }}_{{ ansible_date_time['epoch'] }}.cap

</p>

\2. Then I kicked off a Tcpdump on each server in parallel. (Ansible
runs 5 parallel processes by default, this can be increased by passing
the `--forks=10` parameter to the `ansible-playbook` script.

</p>

<p>
     tasks:    - name: start tcpdump      command: /usr/sbin/tcpdump -i eth0 -s 0 -w /tmp/${cap_file}      async: 60      poll: 0

</p>

As well as running Playbooks against servers in parallel, tasks can be
run asynchronously by setting the `async` (maximum runtime) and `poll`
(how often to poll the job for status). In this example, I'm using
Ansible's [fire-and-forget][] pattern to run the command without
pausing, allowing me to kill it later on in the Playbook.

</p>

\3. Next the script is [paused][] for a minute allowing the testers to
run through their failing tests.

</p>

<p>
        - pause: minutes=1 prompt="pause for 60 seconds or press Ctrl + c then c to continue"

</p>

\4. Then I kill off the tcpdump command with pkill.

</p>

<p>
        - name: kill tcpdump      command: /usr/bin/pkill tcpdump      ignore_errors: yes

</p>

\5. Lastly, I can compress the logs and copy them to my localhost using
the [fetch][] module. The `flat` parameter copies up each file
individually. Without it, the files are stored in a
`${server_name}/{$path}/file` directory structure, ensuring files with
the same name don't overwrite each other.

</p>

<p>
        - name: compress capture file      command: gzip ${cap_file} chdir=/tmp    - name: copy logs to local boxes webroot      fetch: src=/tmp/${cap_file}.gz dest=/var/www/ flat=yes

</p>

It's probably a good idea to clean up the files on the remote server
too.

</p>

<p>
        - name: remove files from server      file: path=/tmp/${cap_file}.gz state=absent

</p>

\6. Now I run it from the command line...

</p>

<p>
    (ENV)> ansible-playbook parallel-tcpdump.yml -i hosts

</p>

And like magic:

</p>

<p>
    (ENV)> ls -1 /var/www/ | grep packet_capturepacket_capture_server1_1376450197.cap.gzpacket_capture_server2_1376450500.cap.gzpacket_capture_server3_1376451234.cap.gz

</p>

Here's the entire Playbook:

</p>

</p>

[And we out][].

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

  [fire-and-forget]: http://www.ansibleworks.com/docs/playbooks2.html#id19
  [paused]: http://www.ansibleworks.com/docs/modules.html#pause
  [fetch]: http://www.ansibleworks.com/docs/modules.html#fetch
  [And we out]: https://twitter.com/lexandstuff
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
