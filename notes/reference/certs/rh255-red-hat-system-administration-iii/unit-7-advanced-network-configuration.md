---
title: Red Hat System Administration III - Unit 7 - Advanced Network Configuration
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## IP Aliasing

* Assign more than one IP address to an interface
    * Unsupported by DHCP
* Can add with `ip addr` command:

        > ip addr show
        1: lo: <LOOPBACK,UP,LOWER_UP> mtu 16436 qdisc noqueue state UNKNOWN
            link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
            inet 127.0.0.1/8 scope host lo
            inet6 ::1/128 scope host
              valid_lft forever preferred_lft forever
        2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP qlen 1000
            link/ether 52:54:00:00:00:02 brd ff:ff:ff:ff:ff:ff
            inet 192.168.0.102/24 brd 192.168.0.255 scope global eth0
            inet6 fe80::5054:ff:fe00:2/64 scope link
              valid_lft forever preferred_lft forever
        > ip addr add 1.1.1.1/24 dev eth0 label eth0:0
        > ip addr show
        1: lo: <LOOPBACK,UP,LOWER_UP> mtu 16436 qdisc noqueue state UNKNOWN
            link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
            inet 127.0.0.1/8 scope host lo
            inet6 ::1/128 scope host
              valid_lft forever preferred_lft forever
        2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP qlen 1000
            link/ether 52:54:00:00:00:02 brd ff:ff:ff:ff:ff:ff
            inet 192.168.0.102/24 brd 192.168.0.255 scope global eth0
            inet 1.1.1.1/24 scope global eth0:0
            inet6 fe80::5054:ff:fe00:2/64 scope link
              valid_lft forever preferred_lft foreve

* To make it persistant, add a file ```/etc/sysconfig/network-scripts/ifcfg-eth0:0```

        > service NetworkManager stop; chkconfig NetworkManager off
        > cat /etc/sysconfig/network-scripts/ifcfg-eth0:0
        DEVICE=eth0:0
        IPADDR=1.1.1.1
        PREFIX=24
        ONPARENT=yes

## Bonding

* To use it, make sure ```NetworkManager``` is off.
* Bonding modes:
    * Mode 0 (balance-rr) - round robin policy; all interfaces are used. Any slave can receive
    * Mode 1 (active-backup) - fault tolerant. Only one slave interface can be used at a time
    * Mode 3 (broadcast) - fault tolerant. All packets are broadcast from the slave interfaces
* Bonding config is stored in: ```/etc/sysconfig/network-scripts/ifcfg-bond0```

        DEVICE=bond0
        IPADDR=1.1.1.1
        PREFIX=24
        ONBOOT=yes
        BOOTPROTO=none
        USERCTL=no
        BONDING_OPTS="mode=1 miimon=50" # Lower miimon, the more checks to the interfaces

* Slave config looks like:

        > cat /etc/sysconfig/network-scripts/ifcfg-<name>
        DEVICE=<name>
        BOOTPROTO=none
        ONBOOT=yes
        MASTER=bond0
        SLAVE=yes
        USERCTL=no

* Create an alias for every bond

        > cat /etc/modprobe.d/bonding.conf
        alias bond0 bonding

* Query state of bonding interface and its slaves using (virtual) file ```/proc/net/bonding/<interface>```

## Tuning Kernel Network Parameters

* ```sysctl``` options are as follows:
    * ```-a``` - list all available params
    * ```-w``` - change a sysctl setting
    * ```-p``` - load in sysctl settings from file
* To setup kernel for packet forwarding:

        > vi /etc/sysctl.conf
        + net.ipv4.ip_forward = 1

* To setup kernel to respond to ICMP broadcase

        > sysctl -w net.ipv4.icmp_echo_ignore_all=0
        > vi /etc/sysctl.conf
        + net.ipv4.icmp_echo_ignore_all = 0

* To view routes (inc default gateway)

        > ip route show
        192.168.0.0/24 dev br0  proto kernel  scope link  src 192.168.0.2
        192.168.122.0/24 dev virbr0  proto kernel  scope link  src 192.168.122.1
        169.254.0.0/16 dev br0  scope link  metric 1004
        default via 192.168.0.254 dev br0

* Add routes with ```ip route add network/netmask via router_ip [dev <interface>]```
* Edit route file with ```/etc/sysconfig/network-scripts/route-iface``` with these 3 lines:

        ADDRESSX=network
        NETMASKX=netmask
        GATEWAYX=router_ip

* ```ping``` uses ICMP to send a request and receives a reply
