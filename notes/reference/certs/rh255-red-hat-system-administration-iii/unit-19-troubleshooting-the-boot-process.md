---
title: Red Hat System Administration III - Unit 19 - Troubleshooting the Boot Process
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## The Boot Process and Rescue Mode

* BIOS
    * What can go wrong
        * Incorrect BIOS settings
    * Troubleshooting
        * Vendor specific
* GRUB
    * Loads initial ram file system ("initramfs")
    * Loads and executes kernel
    * Provides kernel's command-line
    * What can go wrong
        * Misconfiguration of bootloader
        * Bad kernel image or init RAM fs
        * Bad kernel command-line
    * Troubleshooting
        * Choose alternative preconfigured menu item
        * Choose 'e' or 'a' and edit kernel
        * Boot into single-user mode
        * Boot with `init=/bin/bash`
* Kernel
    * Detects hardware devices
    * Loads device drivers
    * Mount the root file system image
    * Start the initial process, init
    * What can go wrong?
        * Bad initial RAM file system image
        * Badly identified root file system
        * Corrupted root file system
* init and Upstart
    * The first userspace process started on the machine is `/sbin/init`. The `init` process is responsible for completing the boot process by starting all other non-kernel processes.
    * Fixing a read only mount:

            > mount -o remount,rw /

* MBR (512 bytes)
    * First 446 bytes - initial boot loader
    * Next 64 bytes - parition table (where is the starting sector and ending sector)
* The Rescue Shell
    * Mount under `/mnt/sysimage`
    * Select rescue media
    * Select the type of network settings
    * Follow prompts
    * Can use yum to reinstall stuff
