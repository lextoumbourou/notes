---
title: Red Hat System Administration III - Unit 18 - File Sharing With FTP
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

* Config: `/etc/vsftpd/vsftpd.comf`
* Anonymous viewing
    * `anonymous_enable = YES`
    * `write_enable` = NO`
    * `local_enable` = NO`
* Dropbox
    * `mkdir /var/ftp/incoming`
    * `chgrp ftp /var/ftp/incoming`
    * `chmod 730 /var/ftp/incoming`

            > ls -ld /var/ftp/incoming/
            drwx-wx---. 2 root ftp 4096 Aug  8 15:26 /var/ftp/incoming/

    * Security context for public ftp directory: t_content_rw_t
    * Adjust boolean:

            > getsebool -a | grep ftpd | grep write
            allow_ftpd_anon_write --> off
            > setsebool -P allow_ftpd_anon_write=on
            > getsebool -a | grep ftpd | grep write
            allow_ftpd_anon_write --> on

    * Fix:

            /etc/vsftpd/vsftpd.conf
            + local_umask=022
            + anon_umask=077
            + chown_uploads=YES
            + chown_username=daemon
            + anonymous_enable = YES
            + write_enable = YES

* Find IPTABLES modules with:

        > find /lib/modules -name '*_ftp*'
        /lib/modules/2.6.32-279.el6.x86_64/kernel/net/ipv4/netfilter/nf_nat_ftp.ko
        /lib/modules/2.6.32-279.el6.x86_64/kernel/net/netfilter/nf_conntrack_ftp.ko
        /lib/modules/2.6.32-279.el6.x86_64/kernel/net/netfilter/ipvs/ip_vs_ftp.ko
        > modinfo nf_conntrack_ftp
        filename:       /lib/modules/2.6.32-279.el6.x86_64/kernel/net/netfilter/nf_conntrack_ftp.ko
        alias:          nfct-helper-ftp
        alias:          ip_conntrack_ftp
        description:    ftp connection tracking helper
        author:         Rusty Russell <rusty@rustcorp.com.au>
        license:        GPL
        srcversion:     8226A329EB50C819ABAC4D7
        depends:        nf_conntrack
        vermagic:       2.6.32-279.el6.x86_64 SMP mod_unload modversions 
        parm:           ports:array of ushort
        parm:           loose:bool
        > modinfo nf_nat_ftp
        filename:       /lib/modules/2.6.32-279.el6.x86_64/kernel/net/ipv4/netfilter/nf_nat_ftp.ko
        alias:          ip_nat_ftp
        description:    ftp NAT helper
        author:         Rusty Russell <rusty@rustcorp.com.au>
        license:        GPL
        srcversion:     F92EE3A32D64466A49CF33B
        depends:        nf_conntrack,nf_nat,nf_conntrack_ftp
        vermagic:       2.6.32-279.el6.x86_64 SMP mod_unload modversions 
