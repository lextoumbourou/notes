---
title: Red Hat System Administration III - Unit 10 - System Monitoring and Logs
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## Usage Reports

* ```df -h``` -- provides info about mounts
* ```du``` -- directory level analysis
    * ```--maxdepth``` -- how many subdirs to traverse
* ```iostat``` -- get details on cpu usage and disk device statistics
    * ```iostat -N``` -- logical volume information
    * ```iostat -dNk <interval> <number_of_repetitions>``` -- logical volume information
* ```vmstat``` -- get virtual memory stats
    * ```r``` -- ??
    * ```b``` -- ??
    * ```swpd``` -- ??
    * ```free``` -- ??
    * ```buff``` -- ??
    * ```cache``` -- ??
    * ```si``` -- ??
    * ```so``` -- ??
    * ```bi``` -- ??
    * ```bo``` -- ??

## Monitoring with AIDE and star

* ```AIDE``` checks the integrity of files on disk

1. Install the ```aide``` package
2. Config ```/etc/aide.conf``` file
3. Run the ```/usr/bin/aide --init``` to build the initial db
4. Store `/etc/aide.conf`, `/usr/bin/aide` and `/var/lib/aide/aide.db.new.gz` in a secure location
5. Copy the `/var/lib/aide/aide.db.new.gz` to `/var/lib/aide/aide.db.gz` (the expected name)

* `sar` is generally used by cron to gather system information about a server
    * Available in `sysstat` package

## Tuning tmpwatch and logrotate

* `tmpwatch` is used to remove folders that haven't been modified in X days
* `logrotate` is used to rotate logs daily

## Remote logging service

* `/etc/rsyslog.conf` file
* Go through man page of `rsyslogd` to get info on facilities levels
    * debug
    * info
    |
    |
    * panic
* To send to remotehost

        > vi /etc/rsyslog.conf
        *.* @@192.168.0.254:514

* Receive syslog messages via TCP:

        # Provides TCP syslog reception
        $ModLoad imtcp
        $InputTCPServerRun 514

## Logrotate

* Force logrotation with:

        > logrotate -f /etc/logrotate.conf
