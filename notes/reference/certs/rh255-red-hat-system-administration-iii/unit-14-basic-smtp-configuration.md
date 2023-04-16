---
title: Red Hat System Administration III - Unit 14 - Basic SMTP Configuration
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

* MTA (Mail Transfer Agent) - relay mail from point to point until it's delivered.
* MDA (Mail Delivery Agent) - deliver mail to recipients local message store. Postfix provised its own MDA to deliver to default local file-based message store: `/usr/libexec/postfix/local`
* MUA (Mail User Agent) - clients used to send email and read email in user's message.

## Key concept

* Relaying - forward from server to server
* Queueing - queued and retrieved periodically by MTA
* Rejected - failed at our end
* Bounced - rejected at their end

## Postfix

* Use `postconf` to see all the available Postfix options
* To make Postfix available on all interfaces, set `inet_interfaces`

        > postconf -e inet_interfaces=all

* `mutt` is a mail client for terminal
* `postfix flush` can clear mail out of mail server for sending
* `/var/spool/mail/$USER` is where a user's mail store is

## Intranet Configuration

* Standard roles:
    * ```null client``` -- runs local MTA, so all email can be forwarded to a central mail server for delivery. Does not accept local delivery for any email messages. Most machines will be null clients.
    * ```inbound```
    * ```outbound```

### Postfix Configuration

* ```inet_interfaces``` - which interfaces to listen on
* ```myorigin``` - usually company domain
* ```relayhost``` - outbound mail server
* ```mydestination``` - company domain on inbound server
    * relevant for inbound!
* ```local_transport``` - for the MDA to put it in the directory
* ```mynetworks``` - (relay from) use for outgoing mail server
    * relevant for outbound!
* Run ```man postconf``` for more info

## Aliases

* Configure an alias for ```andrew``` called ```acctmgr```

        > vi /etc/aliases
        + acctmgr: andrew
        > newaliases

* Questions to answer:
    * Configure your "MTA" mail server according to the following requirements:
        * Your mail server should accept the mail from remote hosts as well as localhost
            * set ```mydestination``` to accept mail for localhost, $myhostname, $mydomain, example.com
