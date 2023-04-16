---
title: Red Hat System Administration III - Unit 16 - File Sharing With NFS 
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

* Port is 2049 (TCP)
* NFS v4
    * Complete redesign of NFS v3
    * Adds: root psuedo fs
    * Supports Kerberos for all implementations
    * Requires no auxillary services
    * Easier to secure behind firewall
    * Use on new deployments where possible
* Syntax to create NFS share
    * Using mount:

            mount <ip_address>:/to/path /mount

    * Improved user mapping support
* Main file: ```/etc/exports```
* How to setup:
    * Create user with same uid on both client and server

            > useradd -u 700 nfs_user

    * Make export directories

            > mkdir -p /exports/{read,write}

    * Ensure owned by user

            > chown nfs_user /exports/write/

    * Create dirs

            > touch /exports/{read,write}/{1,2,3}

    * Edit exports file

            > vi /etc/exports
            /exports/read   *(ro,async)
            /exports/write  182.168.0.0/24(rw,sync,no_root_squash) 

    * Start daemon

            > exportfs -r # Optional, same as service reload
            > service nfs start

* Examples:
    * Export ```/common``` to example.com domain

            > vi /etc/exports
            /common     192.168.0.0/24(rw,async)
            > exportfs -r

    * How to mount

            > vi /etc/fstab
            demo.example.com:/exports/read  /mnt    nfs  ro,vers=4 0 0

    * Client-side options
        * ```rw``` - filesystem is writable
        * ```ro``` - filesystem is read-only
        * ```vers=4``` - try to mount using NFS version specified only.
        * ```soft``` - if NFS request times out, return an error after 3 tries
