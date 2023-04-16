---
title: Red Hat System Administration III - Unit 6 - Network Monitoring
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## netstat

* `-l` - Listening
* `-n` - Numerical hosts
* `-t` - TCP
* `-u` - UDP
* `-p` - Process that controls the port

## nmap

* ```-s``` - statistics
* ```-T``` - stop trimming long addresses

```bash
> nmap -sT demo.example.com

Starting Nmap 5.51 ( http://nmap.org ) at 2013-08-06 12:12 EST
Nmap scan report for demo.example.com (192.168.0.250)
Host is up (0.00043s latency).
Not shown: 997 closed ports
PORT    STATE SERVICE
22/tcp  open  ssh
80/tcp  open  http
111/tcp open  rpcbind
MAC Address: 52:54:00:00:00:FA (QEMU Virtual NIC)
```

* `-P` - ping scan

```bash
> nmap -sP demo.example.com

Starting Nmap 5.51 ( http://nmap.org ) at 2013-08-06 12:14 EST
Nmap scan report for demo.example.com (192.168.0.250)
Host is up (0.00039s latency).
MAC Address: 52:54:00:00:00:FA (QEMU Virtual NIC)
Nmap done: 1 IP address (1 host up) scanned in 0.01 seconds
```

* `-p` - port range

```bash
> nmap -sT -p 100-200 demo.example.com

Starting Nmap 5.51 ( http://nmap.org ) at 2013-08-06 12:14 EST
Nmap scan report for demo.example.com (192.168.0.250)
Host is up (0.00031s latency).
Not shown: 100 closed ports
PORT    STATE SERVICE
111/tcp open  rpcbind
MAC Address: 52:54:00:00:00:FA (QEMU Virtual NIC)

Nmap done: 1 IP address (1 host up) scanned in 0.03 seconds
```

* ```-A``` - enable OS detection, version detection etc

```bash
> nmap -A demo.example.com

Starting Nmap 5.51 ( http://nmap.org ) at 2013-08-06 12:16 EST
Nmap scan report for demo.example.com (192.168.0.250)
Host is up (0.00036s latency).
Not shown: 997 closed ports
PORT    STATE SERVICE VERSION
22/tcp  open  ssh     OpenSSH 5.3 (protocol 2.0)
| ssh-hostkey: 1024 b2:7d:e3:b3:fa:17:04:84:bd:bb:c8:2d:eb:eb:45:70 (DSA)
|_2048 7e:d3:fb:65:c7:cf:54:e7:03:8a:f9:24:d3:72:08:5f (RSA)
80/tcp  open  http    Apache httpd 2.2.15 ((Red Hat))
|_http-title: Test Page for the Apache HTTP Server on Red Hat Enterprise Linux
| http-methods: Potentially risky methods: TRACE
|_See http://nmap.org/nsedoc/scripts/http-methods.html
111/tcp open  rpcbind
MAC Address: 52:54:00:00:00:FA (QEMU Virtual NIC)
No exact OS matches for host (If you know what OS is running on it, see http://nmap.org/submit/ ).
TCP/IP fingerprint:
OS:SCAN(V=5.51%D=8/6%OT=22%CT=1%CU=40507%PV=Y%DS=1%DC=D%G=Y%M=525400%TM=520
OS:05C73%P=x86_64-redhat-linux-gnu)SEQ(SP=105%GCD=1%ISR=10F%TI=Z%CI=Z%II=I%
OS:TS=A)OPS(O1=M5B4ST11NW6%O2=M5B4ST11NW6%O3=M5B4NNT11NW6%O4=M5B4ST11NW6%O5
OS:=M5B4ST11NW6%O6=M5B4ST11)WIN(W1=3890%W2=3890%W3=3890%W4=3890%W5=3890%W6=
OS:3890)ECN(R=Y%DF=Y%T=40%W=3908%O=M5B4NNSNW6%CC=Y%Q=)T1(R=Y%DF=Y%T=40%S=O%
OS:A=S+%F=AS%RD=0%Q=)T2(R=N)T3(R=N)T4(R=Y%DF=Y%T=40%W=0%S=A%A=Z%F=R%O=%RD=0
OS:%Q=)T5(R=Y%DF=Y%T=40%W=0%S=Z%A=S+%F=AR%O=%RD=0%Q=)T6(R=Y%DF=Y%T=40%W=0%S
OS:=A%A=Z%F=R%O=%RD=0%Q=)T7(R=Y%DF=Y%T=40%W=0%S=Z%A=S+%F=AR%O=%RD=0%Q=)U1(R
OS:=Y%DF=N%T=40%IPL=164%UN=0%RIPL=G%RID=G%RIPCK=G%RUCK=G%RUD=G)IE(R=Y%DFI=N
OS:%T=40%CD=S)

Network Distance: 1 hop

TRACEROUTE
HOP RTT     ADDRESS
1   0.36 ms demo.example.com (192.168.0.250)

OS and Service detection performed. Please report any incorrect results at http://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 17.77 seconds
```

## Avahi

* Disable for security reasons
* Use ```netstat``` to see if it's running

```bash
> netstat -lnptu | grep avahi
udp        0      0 0.0.0.0:5353                0.0.0.0:*                               1660/avahi-daemon  
udp        0      0 0.0.0.0:46630               0.0.0.0:*                               1660/avahi-daemon  
```

## Capture and Analyzing

Get all listening interfaces

```bash
> tcpdump -D
1.eth0
2.usbmon1 (USB bus number 1)
3.any (Pseudo-device that captures on all interfaces)
4.lo
```

Capture all ssh network traffic:

* `-nn` - everything (inc ports and protocols as numbers)
* `-l` - do line buffering to file
* `-s` - maximum no of bytes per packet
* `-i` - interface to capture
* `filter` - keywords and logical operators used to filter packets

Example

```bash
> tcpdump -nn -l -s 2000 -w packets -i eth0 'port 22'
> # open packets file in Wireshark
```

Analyse data with Wireshark

```bash
> yum install wireshark-gnome
```
