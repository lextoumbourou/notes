---
title: Asymmetric Cryptography
date: 2024-12-11 00:00
modified: 2026-09-26 08:55
status: draft
---

**Asymmetric Cryptography**, also known as public-key cryptography, is a type of [Cryptography](cryptography.md) that uses a pair of related keys: a public key that can be shared with anyone, and a private key that only the owner has.

A message encrypted with the public key can only be decrypted with the matching private key. See [Public Key Encryption](public-key-encryption.md). It works the other way too: signing a message with the private key lets anyone verify with the public key that it came from the key's owner, which is the basis of digital signatures.

This avoids the main problem with [Symmetric Cryptography](symmetric-cryptography.md), where both parties need to share the same secret key in advance. [RSA](rsa.md) is a well-known example.
