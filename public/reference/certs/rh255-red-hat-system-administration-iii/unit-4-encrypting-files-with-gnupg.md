---
title: Red Hat System Administration III - Unit 4 - Encrypting Files with GnuPG
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## Overview

* Public-key encryption (aka asymmentric encryption)
* Anything encrypted by public key, can only decrypted by private key
* GNU Privacy Guard:
   * GnuPG or GPG is an open-source implementation of OpenPGP

## GPG Commands

1. Generate a key pair:

```gpg --gen-key```

2. List public keys

```gpg --list-keys```

3. Export a public key

```gpg --export --armor -o file.key key-id```

4. Import a public key

```gpg --import key```

5. Encrypt a file

```gpg --encrypt --armor -r <key-id> <file>```

6. Decrypt a file

```gpg --decrpyt <file>```

* Can encrypt with public key and send via to person with private key to decrypt
* Can encrypt with private key and send to people with public key to decrypt
