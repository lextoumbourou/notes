---
title: Red Hat System Administration III - Unit 8 - Secure Network Traffic
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## SSH Port Forwarding

* ```ssh -L <local_port>:<remote_host>:<remote_port> <ssh_host>```
* Example: ```ssh -L 1025:mailhost:25 sshhost```
    * Create a local port on port ```1025```
    * The ssh server on ```sshhost``` forwards traffic to mailhost on port 25
        * Note that the ```sshhost``` will resolve mailhost, not your client machine
* ```ssh -D localport sshhost``` creates a socks proxy through ```sshhost```
    * ```ssh -D 8080 internet_server1```

### Options

* ```-N``` -- don't execute any remote commands (for forwarding ports)
* ```-f``` -- place the ssh connections in the background just before running remote commands

### Preconfigure ssh tunnels

    > vi ~/.ssh/config
    + host instructor
    +    user root
    +    LocalForward 12345 instructor:80
    +    LocalForward 12346 localhost:631 # 631 CUPS administration
    > chmod 600 ~/.ssh/config

## nc

* Basic connectivity tests:

        > nc localhost 25 -z
        Connection to localhost 25 port [tcp/smtp] succeeded!

## iptables

* iptables
    * table (```-t``` or ```--table```)
        * filter (default)
        * nat
        * mangle (ignore)
        * raw
    * chains
        * each table has own set
        * filter = INPUT, OUTPUT, FORWARD
        * nat = PREROUTING, POSTROUTING, OUTPUT
    * rules
    * policy
        * default action to take when nothing passes
        * DROP -- should be default
        * If default is DROP, obviously you'll need to write policy to ACCEPT
    * Targets -- actions to take when packets match rule
        * ACCEPT
        * REJECT -- tell the client we don't want 'em
        * DROP -- don't tell the client
        * LOG
    * Filter
      1. INPUT chain (for anything coming in)
      2. FORWARD chain (send stuff to other systems -- routing)
      3. OUTPUT chain (for anything coming out

### Syntax

    > iptable -nvL --line
    Chain INPUT (policy ACCEPT 184 packets, 22232 bytes)
    pkts bytes target     prot opt in     out     source               destination        

    Chain FORWARD (policy ACCEPT 0 packets, 0 bytes)
    pkts bytes target     prot opt in     out     source               destination        

    Chain OUTPUT (policy ACCEPT 63 packets, 40668 bytes)
    pkts bytes target     prot opt in     out     source               destination        

* here the default ```policy``` is ```ACCEPT```
* To append a rule to the end, use ```-A``` flag.

        > iptables -A INPUT -j REJECT

* To insert a rule at the start (or somewhere else), use ```-I``` flag
* To delete a rule, use ```-D``` flag
* To flush (delete all rules), use ```-F``` flag.
* To accept all traffic from a network:

        > iptables -I INPUT -s 192.168.0.0./24 -j ACCEPT

### Rules (Matching Criteria) Syntax

* Source IP or network

        > -s 192.0.2.0/24

* Destination IP or network

        > -d 10.0.0.1

* UDP/TCP and port

        -p udp --sport 68 --dport 67

### Connection tracking states

* `NEW``
    * packets starts a new connection
    * adds a rule to connection tracking table
* `ESTABLISHED`
    * any packet that matches a rule in the connection tracking table
    * two-way connection like ssh
* `RELATED`
    * like ESTABLISHED but specific to certain services like FTP
* `INVALID`
    * packet cannot be identified
    * normally these should be rejected or dropped
* Another Example
* ```iptables -I INPUT 2 -m state --state ESTABLISHED,RELATED -j ACCEPT```
    * Put into the INPUT chain at the 2nd place
    * For all connections that are ESTABLISHED, RELATED
    * ACCEPT
* Watch packets see where blocked and accepted

```watch -d -n 1 'iptables -nVL --line'```

### Examples

Block all network connectivity to my server except traffic from port 80 and 443

```bash
> iptables -F
> iptables -P INPUT DROP
> iptables -A INPUT -p tcp --dport 80 -j ACCEPT
> iptables -A INPUT -p tcp --dport 443 -j ACCEPT
> service iptables save
```

Allow only SSH connectivity from my desktop to my server

```bash
> DESKTOP=10.0.0.12
> iptables -A INPUT --protocol tcp --dport 22 --source $DESKTOP --jump ACCEPT
> iptables -A INPUT --protocol tcp --source $DESKTOP --jump REJECT
> service iptables save
```

Traffic sent to my **StudyBox** on port 80 should be routed to **XBMC** on port 80

    > XBMC=10.0.0.5
    > iptables --table nat --append PREROUTING \
    --match tcp --protocol tcp \
    --dport 80 \
    --jump DNAT --to-destination $XBMC

Complete firewall policy

    > cat bin/resetfw.sh
    # Set INPUT chain default policy to DROP
    iptables -P INPUT ACCEPT

    # Flush all rule in the filter table
    iptables -F

    # Will ACCEPT all packets from loopback interface
    iptables -A INPUT -i lo -j ACCEPT

    # ACCEPT all ESTABLISHED, RELATED packets
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

    # ACCEPT all NEW connections to tcp port 22
    iptables -A INPUT -m state --state NEW -p tcp --dport 22 -j ACCEPT

    # REJECT all packets from 192.168.1.0/24 network
    iptables -A INPUT -s 192.168.1.0/24 -j REJECT

    # ACCEPT all icmp traffic from 192.168.0.0/24
    iptables -A INPUT -s 192.168.0.0/24 -p icmp -j ACCEPT

    # REJECT all other traffic
    iptables -A INPUT -j REJECT

## Network Address Translation

* Masq causes source ip address to match ip it left firewall on
    * Source aquires public IP of the front-end machine
    * Client -> Firewall (client packet acquires ip of firewall) -> Google
    * Google -> Firewall (firewall knows how to get to client) -> Client
    * Firewall keeps connection in a tracking table
* 3 chains:
    * PREROUTING - processed before routing table
        * DNAT (destination NAT) occurs here
    * OUTPUT
    * POSTROUTING - processed after the routing table
        * SNAT and MASQUERADE occur here
* ```iptables``` rule would look like:

        > iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE

* or

        > iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.0.254
