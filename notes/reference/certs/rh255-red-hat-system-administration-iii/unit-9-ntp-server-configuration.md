---
title: Red Hat System Administration III - Unit 9 - NTP Server Configuration
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## Client side

* ```/etc/ntp.conf```
    * ```server <ntp-server> iburst``` mode allows for quick synchronisation to a server
* ```ntpdate -q instructor.example.com```
* ```ntpd -c pe```

```bash
> ntpq -c pe
remote           refid      st t when poll reach   delay   offset  jitter
==============================================================================
*instructor.exam LOCAL(0)        11 u  483  512  377    0.241   -0.129   0.026
```

* ```jitter``` - 0.000 would mean problem talking to the server
* ```st``` - strata
* ```poll``` - need to research this

## Server side

* Peers/clients
* Each server has a stratum classification - defines how close to reliable time it is (hardware clock is 0) - lower is more accurate
* Clients have higher stratum value
* ```restrict``` option is used to allow access to NTP server
* `server` and `peer` option takes an NTP server and options argument

```bash
> vi /etc/ntp.conf
+ server  instructor.example.com  iburst
+ peer    desktop1.example.com
+ peer    desktop2.example.com
```

* Common flags include:
    * ```ignore``` - deny packets of all kinds, including ntpd and ntpdc queries
    * ```kod``` - Kiss-of-death packets are sent which notifies the client it hit an access violation
    * ```nomodify``` - deny ```ntpq``` and ```ntpdc``` queries whcih attempt to modify the state of the server
    * ```noquery``` - deny ntpq and ntpd queries. Time service is not affected
    * ```nopeer``` - self-explainatory
    * ```notrap``` - decline to provide mode 6 control message

```bash
> vi /etc/ntp.conf
# Hosts on local network are less restricted.
restrict 192.168.1.0 mask 255.255.255.0 nomodify notrap
```

### Example ntp.conf

```bash
#/etc/ntp.conf

restrict default kod nomodify notrap nopeer noquery # Anyone with ipv4
address can query ntp (noquery != ignore)
restrict -6 default ignore # no one with ipv6 can query NTP

restrict 192.168.0.0 mask 255.255.255.0 nomodify notrap nopeer
restrict 192.168.0.101 kod nomodify notrap
restrict 192.168.0.200

server 192.168.0.2
server 192.168.0.3
peer 192.168.0.101
```
