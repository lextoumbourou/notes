Title: Getting Started With Ansible
Slug: getting-started-with-ansible
Date: 2013-02-04

[Overview][]  

[Why Ansible?][]  

[My Requirements][]  

[Part 1: Preparing my Inventory][]  

[Part 2: Configuring Users and Groups with Modules][]  

[Part 3: Automating Tasks Using Playbooks][]  

[Part 4: Using Templates To Setup Privileges and SSH Security][]  

[Part 5: Source Control][]  

[Summary][]  

</p>

* * * * *

</p>

<a name="overview"></a>

### Overview

</p>

This weekend I decided to fix the mess that was my side-project
production environment. I had 3 Linode VMs running 3 or 4 production
websites and a couple of websites in development scattered haphazardly.
There was a Postgres instance wasting cycles on each and no unity across
configs. So, I decided I was going to implement some form of automated
configuration management. Tools like Puppet or Chef seemed like
reasonable options, especially since we use Puppet at work, but I have
been wanting to play with Ansible for a while and this seemed like a
good opportunity.

</p>

<a name="why-ansible"></a>

### Why Ansible?

</p>

In short: simplicity. Ansible is the kind of tool that you just pick up
and use; the learning curve is minimal. All the communication is handled
via vanilla SSH so there's no clients to install and it uses a push
system for deploying changes, so no central server is required. This
makes the configs portable and the setup costs low. The Playbooks (like
Cookbooks in Chef) are written in YAML and the template engine uses
Jinja2 - tools I'm very familiar with. Lastly, and perhaps for me the
biggest draw card, the source code is written in Python.

</p>

<a name="requirements"></a>

### My Requirements

</p>

Before beginning my Ansible journey, I set aside some goals to guide the
process:

</p>

1.  I should never, *ever* have to log into a production server to
    configure it. That means everything down to the first user account
    should be handled by Ansible.
2.  Everything configured should be completely self documenting. If I
    come back to them after a year, it should be very clear what
    everything does.
3.  The entire system should be extremely portable. Any time I need to
    rebuild my development server, the configs should just be a
    `git pull` away.</a>
    </p>
    <p>

</p>

Which that in mind, I documented the steps I followed to get the basics
up and running: user accounts, groups, sshd\_config and sudoers. I have
been able to get much further than this in a very short period of time
(thanks largely to Ansible's excellent documentation and ease of use),
in the interest of brevity, let's keep it...brief.

</p>

* * * * *

</p>

<a name="part-1"></a>

### Part 1: Preparing my Inventory

</p>

I started by creating a new Ubuntu VM at Linode and setting a default
password.

</p>

Then, on my development machine, I installed Ansible's prerequisties:
[PyYAML][], [Jinja2][] and [Paramiko][].

</p>

<p>
    lex@desktop:~> sudo pip install pyyaml jinja2 paramiko

</p>

*Note: it is trivial to run Ansible in a [virtualenv][] container and
generally a good idea. For simplicity sake, I've skipped those steps
here.*

</p>

Next, following Ansible's [install instructions][], I cloned the repo,
checking out the latest stable branch and installed it using
[distutils][].

</p>

<p>
    lex@desktop:~> cd srclex@desktop:~/src> git clone -b release0.9 git://github.com/ansible/ansible.gitlex@desktop:~/src> cd ansiblelex@desktop:~/src/ansible> sudo python setup.py install

</p>

After, I created the `/etc/ansible/` directory which housed my first
[hosts][] file, initially just listing my created VM (but would soon
list all my servers in production).

</p>

##### /etc/ansible/hosts

</p>
<p>
    [webservers]prodweb03

</p>

The host file can also include additional per-host and per-group
variables which are useful for configuring things like network settings
and so on. Consult the [Inventory and Patterns][] section of the doco
for more info.

</p>

I then tested connectivity to the web servers using the `ping` module
(more on Ansible's modules and syntax later).

</p>

<p>
    lex@desktop:/etc/ansible> ansible all -m ping -u root --ask-passSSH password:prodweb03 | success >> {    "changed": false,    "ping": "pong"}

</p>

<a name="part-2"></a>

### Part 2: Configuring Users and Groups with Modules

</p>

Ansible has a reasonable library of modules built-in which can be
executed via Playbooks or directly via the command line. The latter's
[syntax][] looks a bit like this:

</p>

<p>
    lex@desktop:~> ansible <host_pattern> -m <module_name> -a <module_args> <additional_args>

</p>

For example, I can utilise the `command` module to run `uptime` across
my hosts.

</p>

<p>
    lex@desktop:/etc/ansible> ansible webservers -m command -a "uptime" -u root --ask-passSSH password:prodweb03 | success | rc=0 >> 06:55:23 up 12 min,  1 user,  load average: 0.00, 0.01, 0.01

</p>

*Note that by default, Ansible attempts to perform operations as the
local user, using public key authentication. Since I don't have any
users setup remotely just yet, nor any keys, I'll have to explicitly run
the command as root (`-u root`) forcing a password prompt
(`--ask-pass`).*

</p>

To start the configuration process, I configured my user accounts using
the `group` and `user` modules, with the intention of later moving the
commands to Playbooks.

</p>

Here I created a group called `admin`, with a GID of 1000.

</p>

<p>
    lex@desktop:/etc/ansible> ansible webservers -m group -a "gid=1000 name=admin state=present" -u root --ask-pass

</p>

Hopefully, the arguments I'm passing to the module are self-explanatory.
If not, consult the amazingly simple [documentation][].

</p>

Next, I created my user account with the `user` module, and uploaded my
`authorized_key` file using the module with the same name.

</p>

<p>
    lex@desktop:/etc/ansible> ansible webservers -m user -a \                          "name=lex group=admin shell=/bin/bash uid=1000" -u root --ask-passlex@desktop:/etc/ansible> ansible webservers -m authorized_key -a \                          "user=lex key='$(cat ~/.ssh/id_rsa.pub)'" -u root --ask-pass

</p>

Now, since I have `ssh-agent` running, I can perform commands across the
server(s) as myself.

</p>

<p>
    lex@desktop:/etc/ansible> ansible webservers -m command -a "uptime"prodweb03 | success | rc=0 >> 07:31:47 up 48 min,  1 user,  load average: 0.00, 0.01, 0.02

</p>

<a name="part-3"></a>

### Part 3: Automating Tasks Using Playbooks

</p>

Playbooks are [YAML][] configuration files that define a set of tasks or
*Plays* to be executed on the remote server. Following Ansible's [Best
Practises][] document, I created a `company_name` directory to house my
Playbooks and additional files. Then, I created the first Playbook,
`initial.yml`, where I would store the user and group creation tasks I
ran manually earlier.

</p>

<p>
    lex@desktop:/etc/ansible> mkdir lexandstufflex@desktop:/etc/ansible> cd lexandstuff/lex@desktop:/etc/ansible/lexandstuff> vi initial.yml

</p>

##### /etc/ansible/lexandstuff/initial.yml

</p>
<p>
    - hosts: webservers  sudo: yes  tasks:    # ===============================================================    # Configure user's and groups    # ===============================================================    - name: create admin group      action: group name=admin gid=1000 system=no    - name: create user(s)      action: user name=lex group=admin shell=/bin/bash uid=1000    - name: setup authorized key(s)      action: authorized_key user=lex key='$FILE(/home/lex/.ssh/id_rsa.pub)'

</p>

The syntax is quite self-explanatory: `hosts` represents the group of
servers this Playbook will refer to. `sudo` defines whether or not we'll
run the commands using sudo. Then, in the `tasks` section, `name` is the
human-readable task description that appears on the command line while
the Playbook is running. Clearly then, `action` represents the actual
task.

With that complete, I can now run the Playbook at the prompt and watch
the magic happen.

</p>

<p>
    lex@desktop:/etc/ansible/lexandstuff> ansible-playbook initial.yml -u root --ask-passSSH password:PLAY [webservers] *********************GATHERING FACTS *********************ok: [prodweb03]TASK: [create group(s)] *********************ok: [prodweb03]TASK: [create user(s)] *********************ok: [prodweb03]TASK: [setup authorized key(s)] *********************ok: [prodweb03]PLAY RECAP *********************prodweb03                      : ok=4    changed=0    unreachable=0    failed=0

</p>

Perhaps at some stage I expect to require more than just a single
account and user group on my production web servers. One way to handle
it, is to just list additionally actions in order.

</p>

##### /etc/ansible/lexandstuff/initial.yml

</p>
<p>
        #...    - name: create admin group      action: group name=admin gid=1000 system=no    - name: create sshusers group      action: group name=sshusers gid=1001 system=no    #...

</p>

Or, I could separate a repeatable task into a [separate file][] and call
it from within my main Playbook.

</p>

##### /etc/ansible/lexandstuff/initial.yml

</p>
<p>
       #...    - include: tasks/create_user.yml user=lex group=admin full_name='Lex Toumbourou' uid=1000    - include: tasks/create_user.yml user=travis group=sshusers full_name='Travis Bickle' uid=1001    - include: tasks/create_user.yml user=jacob group=sshusers full_name='Jacob Singer' uid=1002   #...

</p>

##### /etc/ansible/lexandstuff/tasks/create\_user.yml

</p>
<p>
    - name: create user(s)  action: user name=$user group=$group shell=/bin/bash uid=$uid- name: setup authorized key(s)  action: authorized_key user=$user key='$FILE(/home/$user/.ssh/id_rsa.pub)'

</p>

<a name="part-4"></a>

### Part 4: Using Templates To Setup Privileges and SSH Security

</p>

Ansible relies on the very powerful Jinja2 engine to handle templating.
In this last part of the tutorial, I'm going to utilise templates to
create a custom sudoers file, allowing me to run commands on the remote
hosts as myself, and a customised sshd config.

</p>

Firstly, I created the templates directory and wrote the custom sudoers
template. You may note that initially, I'm not taking advantage of the
templating engine at all. I could have simply used the `copy` module to
copy the file up to my server, but this gives me a chance to extend it
later.

</p>

##### /etc/ansible/lexandstuff/templates/custom\_sudo.j2

</p>

<p>
    # Members of the admin group may gain root privileges%admin ALL=(ALL) NOPASSWD:ALL

</p>

Now, I can import add the creation of `/etc/sudoers.d/custom` into the
Playbook.

</p>

##### /etc/ansible/lexandstuff/initial.yml

</p>
<p>
      #...  tasks:    #...    # ===============================================================    # Access, security and permissions    # ===============================================================    - name: write the sudoers file      action: template src=templates/sudoers.j2 dest=/etc/sudoers.d/custom                 owner=root group=root mode=0400   #...

</p>

Now, for my `sshd_config`, I created another template with placeholders
for the variables configured in the Playbook, using Jinja2's
`{{ variable_name }}` syntax.

</p>

##### /etc/ansible/lexandstuff/templates/sshd\_config.j2

</p>
<p>
    Port {{ ssh_port }}Protocol {{ ssh_protocol }}#...RSAAuthentication {{ ssh_rsa_authentication }}PubkeyAuthentication {{ ssh_public_key_authentication }}#...

</p>

And, so on.

</p>

In a separate `vars` file, I specified the variables...

</p>

##### /etc/ansible/lexandstuff/vars/defaults.yml

</p>
<p>
    ######################### sshd_config settings########################ssh_port: 22ssh_protocol: 2ssh_syslog_facility: AUTH#...# Allow authentication methodsssh_rsa_authentication: yesssh_public_key_authentication: yes#...

</p>

...which were included in the Playbook using the `vars_files` parameter.

</p>

##### /etc/ansible/lexandstuff/initial.yml

</p>
<p>
    # ...- hosts: all  sudo: yes  gather_facts: no  vars_file:    - vars/defaults.yml   tasks:# ...

</p>

Then, I just call the `template` module to create the `sshd_config`
file.

</p>

##### /etc/ansible/lexandstuff/initial.yml

</p>
<p>
    # ...    - name: write the sshd_config file      action: template src=templates/sshd_config.j2 dest=/etc/ssh/sshd_config              owner=root group=root mode=0644

</p>

In order for the changes to take effect, we're going to need to reload
the ssh daemon after changing the config. That can be done with a
`handler`. A handler is a list of tasks that another task can call or
"notify" to perform after the Playbook has finished executing. So, in
the config, I've added a handler section toward the bottom which uses
the `service` module to reload the config.

</p>

##### /etc/ansible/lexandstuff/initial.yml

</p>
<p>
    #...  handlers:    - name: reload sshd      action: service name=ssh state=reloaded

</p>

Then, using the `notify` parameter, I can call the handler after the
sshd config file is generated.

</p>

##### /etc/ansible/lexandstuff/initial.yml

</p>
<p>
        #...    - name: write the sshd_config file      action: template src=templates/sshd_config.j2 dest=/etc/ssh/sshd_config              owner=root group=root mode=0644      notify:        - reload sshd

</p>

And we're done.

</p>

<a name="part-5"></a>

### Part 5: Source Control

</p>

Lastly, I put my entire repo in source control.

</p>

<p>
    lex@desktop:/etc/ansible/lexandstuff> git initInitialized empty Git repository in /etc/ansible/lexandstuff/.git/lex@desktop:/etc/ansible/lexandstuff> git add .lex@desktop:/etc/ansible/lexandstuff> git commit -m "First working version configures basic environment"

</p>

I then pushed it to a private repo on GitHub, which means that I have
access to them from anywhere.

</p>

<a name="summary"></a>

### Summary

</p>

In conclusion, Ansible is awesome. Straight-forward, easy,
self-documenting and, dare I say it, fun. If you're finding yourself
putting off automating configuration management because you're fearing a
steep learning curve, then Ansible is probably for you.

</p>

* * * * *

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

  [Overview]: http://feeds.feedburner.com/lextoumbourou#overview
  [Why Ansible?]: http://feeds.feedburner.com/lextoumbourou#why-ansible
  [My Requirements]: http://feeds.feedburner.com/lextoumbourou#requirements
  [Part 1: Preparing my Inventory]: http://feeds.feedburner.com/lextoumbourou#part-1
  [Part 2: Configuring Users and Groups with Modules]: http://feeds.feedburner.com/lextoumbourou#part-2
  [Part 3: Automating Tasks Using Playbooks]: http://feeds.feedburner.com/lextoumbourou#part-3
  [Part 4: Using Templates To Setup Privileges and SSH Security]: http://feeds.feedburner.com/lextoumbourou#part-4
  [Part 5: Source Control]: http://feeds.feedburner.com/lextoumbourou#part-5
  [Summary]: http://feeds.feedburner.com/lextoumbourou#summary
  [PyYAML]: http://pyyaml.org/
  [Jinja2]: http://jinja.pocoo.org/docs/
  [Paramiko]: http://www.lag.net/paramiko/
  [virtualenv]: http://www.virtualenv.org/en/latest/
  [install instructions]: http://ansible.cc/docs/gettingstarted.html#contents
  [distutils]: http://docs.python.org/2/distutils/index.html
  [hosts]: http://ansible.cc/docs/patterns.html#hosts-and-groups
  [Inventory and Patterns]: http://ansible.cc/docs/patterns.html
  [syntax]: http://ansible.cc/docs/examples.html
  [documentation]: http://ansible.cc/docs/modules.html
  [YAML]: http://www.yaml.org/
  [Best Practises]: http://ansible.cc/docs/bestpractices.html
  [separate file]: http://ansible.cc/docs/playbooks.html#include-files-and-encouraging-reuse
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
