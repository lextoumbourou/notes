---
title: Red Hat System Administration III - Unit 17 - File Sharing With CIFS
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

* Common Internet File Sysmte formerly known as SMB (Server Message Block)
* Access a CIF share with `//hostname/share`
* Mount via the GUI with **Applications** > **System Tools** > **File Browser**
* Via the command-line:

        > smbclient //instructor/ftp -U guest2002%optional_password

    * `-L` - options lists
* Via `mount` command:

        > mount -t cifs -o user=guest20012,pass=password //instructor.example.com/ftp /mnt/place

* Via `fstab`:

        > vi /etc/fstabl
        + //instructor/ftp  /mnt    cifs    credentials=/root/password.txt 0 0
        vi /root/password.txt
        + user=username
        + pass=password
        + domain=WHATEVS

## Providing Home Directories as CIFS Shares

* Install parts of Samba service:

        > yum -y install samba samba-common samba-client samba-doc

* Allow access:

        > vi /etc/samba/smb.conf
        + hosts allow = 127. 192.168.0.
        + security = user

* Start Samba and put on startup

        > service smb start
        > chkconfig smb on

* Security types
    * ```security = user``` - use local passwd file
    * ```security = share``` - open to all
    * ```security = domain``` - needs to be in NT domain
    * ```security = server``` - pass to another SAMBA server for auth
    * ```security = ads``` - Active Directory
* Share definitions

        [homes]
        workgroup = SAMBAGRP
        name = 

* Add user
    * Add user to password file

     ```useradd -s /sbin/nologin blahuser```

    * Create a Samba user in ```/var/lib/samba/private/passdb.db

            > smbpasswd -a blahuser
            > pdbedit -L
            blah:701:

    * Get Bools for Samba

        > getsebool -a | grep samba
        > setsebool -P samba_enable_home_dirs on

    * Set firewall rules
        * UDP: 137 : 138
        * TCP: 139, 445

                iptables -P INPUT DROP
                iptables -A INPUT -p tcp --dport 445 -j ACCEPT
                iptables -A INPUT -p tcp --dport 139 -j ACCEPT
                iptables -A INPUT -p udp --dport 137:138 -j ACCEPT
                iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

* CIFS settings for shared directory

        > chmod 2770 ${groupdir}
        -----------------------
        [${sharename}]
        path=${groupdir}
        valid users = @${groupname}
        readonly = no or writeable = yes
        public = no or guest ok = no
        -------------------------
        chmod 227 ${groupdir}
        -------------------------
        [${sharename}]
        path = ${groupdir}
        read only = yes or writeable = no
        write list = @${groupname}
        public = no or guest ok = no
        ------------------------------
        CIFS SETTINGS FOR PRINT SHARE
        [${printershare}]
        comment = ${printer_description}]
        path = /var/spool/samba
        read only = yes
        printable = yes
        printer name = ${printername}

* `semanage fcontent -a -t samba_share_t '/shared/school(/.*)?'`
* `restorecon -VFR /testing@wheel`

## Examples

* Share the directory /command via SAMBA. Your SAMBA server must be a member of the SAMBAGRP workgroup.
* The share name must be "comment".
* The shared must be available to example.com clients only.
* The user "natasha" should have read access to the share with the SAMBA password "redhat".

        > yum -y samba samba samba-client samba-doc samba-config
        > # To do
