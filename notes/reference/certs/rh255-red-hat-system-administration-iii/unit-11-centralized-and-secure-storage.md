---
title: Red Hat System Administration III - Unit 11 - Centralized and Secure Storage
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## Recap of setting up local disks

* Create partitions:

        > fdisk -cu <device_name>

    * Always reboot after creating parition
* Create a filesystem:

        > mkfs.ext4 /dev/sdb1

* Create a mount point

        > mkdir /mymount

* Mount partition

        > mount /mymount /dev/sdb1

* Create an entry in ```/etc/fstab```

        > vi /etc/fstab
        mount  dev_name  fstype  options  <dump_freq>  <fsck 0/1/2>

## Accessing iSCSI Storage

* Targets == Servers
* Initators == Clients
* iSCSI Qualified Name (IQN) identifies initiators and targets

        ign.yyyy-mm.{reverse domain}:label

### Example

* Install prereqs:

        > yum install iscsi-initator-utils

* Discover iSCSI targets

        > iscsiadm -m discovery -t st -p 192.168.0.254
        192.168.0.254:3260,1 iqn.2010-09.com.example:rdisks.server2

* Log on to one or more iSCSI targets

        > iscsiadm -m node -T iqn.2010-09.com.example:rdisks.server2 -l
        Logging in to [iface: default, target: iqn.2010-09.com.example:rdisks.server2, portal: 192.168.0.254,3260] (multiple)
        Login to [iface: default, target: iqn.2010-09.com.example:rdisks.server2, portal: 192.168.0.254,3260] successful.

* Check status

        > ls -l /dev/disk/by-path/*iscsi*
        lrwxrwxrwx. 1 root root 9 Aug  7 13:36 /dev/disk/by-path/ip-192.168.0.254:3260-iscsi-iqn.2010-09.com.example:rdisks.server2-lun-1 -> ../../sda

* So, I know that it's using sda as a physical-like disk:

        > fdisk /dev/sda

* Get UUID of device with:

        > blkid | grep sda
        /dev/sda1: UUID="3fa18c35-17bd-4eb3-9409-9a7f211ea8d7" TYPE="ext4"

* Use UUID to mount within ```/etc/fstab```

        > vi /etc/fstab
        + UUID="3fa18c35-17bd-4eb3-9409-9a7f211ea8d7"     /mnt/iscsi      ext4    _netdev 0 0

* Then mount it!

        > mkdir /mnt/iscsi
        > mount /mnt/iscsi

### To remove

* Log out of iSCSI target

        > iscsiadm -m node -T iqn.2010-09.com.example:rdisks.server2 -p 192.168.0.254 -u
        Logging out of session [sid: 1, target: iqn.2010-09.com.example:rdisks.server2, portal: 192.168.0.254,3260]
        Logout of [sid: 1, target: iqn.2010-09.com.example:rdisks.server2, portal: 192.168.0.254,3260] successful.

* Delete the local record of the iSCSI targer

        > iscsiadm -m node -T iqn.2010-09.com.example:rdisks.server2 -p 192.168.0.254 -o delete

## Encrypt Centralized Storage

* Basic procedure:
    * Encrypt device with `cryptsetup luksFormat`:

            # cryptsetup luksFormat /dev/sda1

    * Open the device with `cryptsetup luksOpen` and set a passphrase:

            # cryptsetup luksOpen /dev/sda1 secret

    * Create a filesystem:

            # mkfs.ext4

    * Get the UUID with ```blkid```:

            # blkid | grep /dev/mapper/secret

    * Add an entry to crypttab:

            # vi /etc/crypttab
            UUID=""     /dev/mapper/secret      (none) or (path-to-passphrase)

    * Add entry in ```/etc/fstab```:

            > vi /etc/fstab
            /dev/mapper/secret  /mnt/my_secret  file_system_type _netdev  0 0
