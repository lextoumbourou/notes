---
title: Red Hat System Administration III - Unit 2 - Enable User Security
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## Configuring sudo

* Basic syntax:

        USER MACHINE = (RUN_AS) COMMANDS

* Examples:
    * `%group ALL = (root) /sbin/ifconfig`
    * `%wheel ALL=(ALL) ALL`
    * `%admins ALL=(ALL) NOPASSWD: ALL`
    * Grant use access to commands in NETWORKING group
        * `joseph ALL=NETWORKING`
* Use `User_Alias` to grant access to list of users:

    User_Alias ADMINS = lex, john

* Use `Cmnd_Alias` to set a list of commands:

    Cmnd_Alias NETWORKING = /sbin/ruote, /sbin/ifconfig

* Aliases are "string of uppercase letters, numbers, and the underscore characters"
* Groups are denoted with `%group, username, #uid`:

        # man sudoers
        Runas_Member ::= '!'* user name |
                            '!'* '#'uid |
                            '!'* '%'group |
                            '!'* +netgroup |
                            '!'* Runas_Alias

## Kerberos Authentication

* Kerberos more secure than LDAP as it doesn't pass passwords over network
* How it works:
    * KDC - key distribution center
        * grants tickets
    * KAdmin Server
        * used to update passwords
    * Realm
        * set of systems which use the same KDCs
* Configuring auth:
    * Via GUI: ```system-config-authentication```
    * Via command-line: ```authconfig```
    * Question: Which command line tool can be used for configuring Kerberos?
    * Answer: authconfig

            # authconfig --enableldap --ldapserver=instructor.example.com \
                --enableldaptls --ldaploadcacert=ftp://instructor.example.com/pub/example-ca.crt \
                --ldapbasedn="dc=example,dc=com" --disableldapauth --enablekrb5 \
                --krb5kdc=instructor.example.com --krb5adminserver=instructor.example.com \
                --krb5realm=EXAMPLE.COM --enablesssdauth --update

* Packages required:
    * ```yum groupinstall directory-client```
    * ```yum install openldap-clients```
    * ```yum install krb5-workstation```
* Kerberos commands:
    * ```klist``` - list tickets
    * ```kdestroy``` - delete tickets
    * ```kinit``` - get new tickets
    * Example:

            # klist
            Ticket cache: FILE:/tmp/krb5cc_1701_5hTCSt
            Default principal: ldapuser1@EXAMPLE.COM

            Valid starting     Expires            Service principal
            08/05/13 12:35:36  08/06/13 12:35:35  krbtgt/EXAMPLE.COM@EXAMPLE.COM
                renew until 08/05/13 12:35:36
            $ kdestroy
            $ klist
            klist: No credentials cache found (ticket cache FILE:/tmp/krb5cc_1701_5hTCSt)
            $ kinit
            Password for ldapuser1@EXAMPLE.COM:
            $ klist
            Ticket cache: FILE:/tmp/krb5cc_1701_5hTCSt
            Default principal: ldapuser1@EXAMPLE.COM

            Valid starting     Expires            Service principal
            08/05/13 12:35:46  08/06/13 12:35:44

* Offline LDAP and Kerberos servers can prevent login, but sssd can cache credentials allowing offline login:
    * Edit the ```/etc/sssd/sssd.conf``` to configure SSSD
    * Use ```authconfig``` to edit it via the command-line
    * Service could be cached hiding a downed server
    * Logging:
        * `/var/log/sssd`
        * Increase verbosity in `/etc/sssd/sssd.conf` by adding `debug_level=10` under `[domain/default]` (0 - 10)
