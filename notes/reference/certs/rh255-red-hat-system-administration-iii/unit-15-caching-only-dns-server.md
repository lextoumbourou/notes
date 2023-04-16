---
title: Red Hat System Administration III - Unit 15 - Caching Only DNS Server
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## DNS General

* Authoritative
    * Master - contains original zone data. A "primary" nameserver
    * Slave - backup server. Copy of zone data.
* Non-authoritative / recursive nameservers
    * Caching-only - used for lookups, not authoritative for anything cept trivial data
* Enable 53 for TCP and UDP for DNS

## Bind

* Package `bind`
* config = `/etc/named.conf`

        > vi /etc/named.conf
        + listen-on port 53 { any; };
        + listen-on-v6 port 53 { any; };
        + allow-query     { 192.168.0.0/24; };
        + dnssec-validation no;  // stops spoofing DNS

* Dump contents of caching db

        > rndc dumpdb --cache
        > cat /var/named/data/cache_dump.db

## DNS Lookups

* Stub resolve sends query to client in ```/etc/resolv.conf```
* If nameserver is authoritative, send authoritative answer to client
    * A - forward lookup (name to ipv4 address)
    * AAAA - forward lookup (name to ipv6 address)
    * PTR - reverse record
    * MX - mail exchange
    * SOA - start of authoritity
    * CNAME - alias (canonical name)
* Use ```dig``` for troubleshooting

        > dig www.example.com

        ; <<>> DiG 9.8.2rc1-RedHat-9.8.2-0.10.rc1.el6 <<>> www.example.com
        ;; global options: +cmd
        ;; Got answer:
        ;; ->>HEADER<<- opcode: QUERY, status: NXDOMAIN, id: 52706
        ;; flags: qr aa rd ra; QUERY: 1, ANSWER: 0, AUTHORITY: 1, ADDITIONAL: 0

        ;; QUESTION SECTION:
        ;www.example.com.       IN  A

        ;; AUTHORITY SECTION:
        example.com.        60  IN  SOA instructor.example.com. root.instructor.example.com. 2010091500 3600 300 604800 60

        ;; Query time: 0 msec
        ;; SERVER: 192.168.0.254#53(192.168.0.254)
        ;; WHEN: Thu Aug  8 11:29:47 2013
        ;; MSG SIZE  rcvd: 85
