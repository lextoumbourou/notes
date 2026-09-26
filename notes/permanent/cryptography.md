---
title: Cryptography
date: 2024-12-11 00:00
modified: 2026-09-26 08:55
status: draft
---

**Cryptography** is the study of communicating in secret using data [Encryption](encryption.md) and [Decryption](decryption.md). The word has Greek roots: *crypt* (hidden/secret) and *graphy* (written).

In cryptography, we typically define three metaphorical parties: Bob and Alice, who want to communicate together securely, and Eve, who wants to either listen in, modify or pretend to be them.

We call the original, unencrypted message that Bob and Alice want to send to each other [Plaintext](plaintext.md) and the encrypted message [Ciphertext](ciphertext.md). The process of converting between Plaintext and Ciphertext is called [Encryption](encryption.md), and the reverse, [Decryption](decryption.md).

There are two key methods of encrypting a message:

1. [Symmetric Cryptography](symmetric-cryptography.md), where the same key is shared by both Bob and Alice. This is also known as [Symmetric Cryptography](symmetric-cryptography.md). Think of this as locking a box, and both Bob and Alice have keys to the box.

2. [Public Key Encryption](public-key-encryption.md), where we have two keys: one used to encrypt a message, and another to decrypt it. In the box example, Bob can make copies of the padlock, but only he has the key. Alice can lock a message with a padlock, knowing that only Bob has the key.

The 4 main goals of cryptography are:

* **Secrecy/Confidentiality**: Preventing unauthorized access to information
* **Integrity**: Ensuring messages cannot be altered without detection
* **Authentication**: Verifying the sender's identity
* **Non-repudiation**: Preventing senders from denying they sent a message