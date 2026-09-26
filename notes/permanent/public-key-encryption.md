---
title: Public Key Encryption
date: 2025-03-30 00:00
modified: 2026-09-26 08:55
status: draft
---

**Public Key Encryption** is a type of [Encryption](encryption.md) that uses a pair of keys: a public key, which can be shared with anyone and is used to encrypt, and a private key, which is kept secret and used to decrypt.

Anyone can send you a message encrypted with your public key, but only you can read it. This solves the key distribution problem of [Symmetric Cryptography](symmetric-cryptography.md), where both sides need to share the same secret key.

[RSA](rsa.md) is a well-known example. See [Asymmetric Cryptography](asymmetric-cryptography.md).
