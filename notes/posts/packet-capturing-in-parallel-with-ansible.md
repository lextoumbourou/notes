---
title: Packet capturing in parallel with Ansible
date: 2013-08-13 00:00
slug: packet-capturing-in-parallel-ansible
---

<a name="intro"></a>

<div class="intro">
Ansible continues to amaze me. It seems to be able to solve so many
problems with such simplicity and I keep finding new uses for it. Enter
today's problem and solution.
</div>

### The Problem

* Run Tcpdump for a short window in parallel across a massive test
    environment while front-end tests are performed
* Gzip and tar each generated pcap and copy to a remote share for
    distribution

After getting my environment setup right, the solution took me a total
of 30 minutes to write and deploy. Plus, it is completely reusable in
any environment (though the interface parameter passed to tcpdump may
require tweaking).

### The Solution

1\. First I utilised Ansible's host variables to create a unique file
name for each server.

    :::yaml
    - hosts: all
      sudo: yes
      vars:
          cap_file: packet_capture_{{ ansible_hostname }}_{{ ansible_date_time['epoch'] }}.cap

2\. Then I kicked off a Tcpdump on each server in parallel. Ansible
runs 5 parallel processes by default, this can be increased by passing
the `--forks=10` parameter (replacing 10 with the number of servers) to the `ansible-playbook` script.

    :::yaml
      tasks:
         - name: start tcpdump
           command: /usr/sbin/tcpdump -i eth0 -s 0 -w /tmp/${cap_file}
           async: 60
           poll: 0

As well as running Playbooks against servers in parallel, tasks can be
run asynchronously by setting the `async` (maximum runtime) and `poll`
(how often to poll the job for status). In this example, I'm using
Ansible's [fire-and-forget][] pattern to run the command without
pausing, allowing me to kill it later on in the Playbook.

3\. Next the script is [paused][] for a minute allowing the testers to
run through their failing tests.

    :::yaml
        - pause: minutes=1 prompt="pause for 60 seconds or press Ctrl + c then c to continue"

4\. Then I kill off the tcpdump command with pkill.

    :::yaml
        - name: kill tcpdump
          command: /usr/bin/pkill tcpdump
          ignore_errors: yes

5\. Lastly, I can compress the logs and copy them to my localhost using
the [fetch][] module. The `flat` parameter copies up each file
individually. Without it, the files are stored in a
`${server_name}/{$path}/file` directory structure, ensuring files with
the same name don't overwrite each other.

    :::yaml
        - name: compress capture file
          command: gzip ${cap_file} chdir=/tmp
     
        - name: copy logs to local boxes webroot
          fetch: src=/tmp/${cap_file}.gz dest=/var/www/ flat=yes

It's probably a good idea to clean up the files on the remote server
too.

    :::yaml
        - name: remove files from server
          file: path=/tmp/${cap_file}.gz state=absent

6\. Now I run it from the command line...

    :::bash
    (ENV)> ansible-playbook parallel-tcpdump.yml -i hosts

And like magic:

    :::bash
    (ENV)> ls -1 /var/www/ | grep packet_capture
    packet_capture_server1_1376450197.cap.gz
    packet_capture_server2_1376450500.cap.gz
    packet_capture_server3_1376451234.cap.gz

Here's the entire Playbook:

<script src="https://gist.github.com/lextoumbourou/7611499.js"></script>

  [fire-and-forget]: http://www.ansibleworks.com/docs/playbooks2.html#id19
  [paused]: http://www.ansibleworks.com/docs/modules.html#pause
  [fetch]: http://www.ansibleworks.com/docs/modules.html#fetch
  [And we out]: https://twitter.com/lexandstuff
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
