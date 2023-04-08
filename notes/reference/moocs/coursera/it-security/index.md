---
title: index
date: 2023-03-25 00:00:00
category: reference/moocs
slug: it-security
summary: "Notes from [IT Security: Defense against the digital dark arts](https://www.coursera.org/learn/it-security)"
status: draft
modified: 2023-04-08 00:00:00
---

## Week 1

### CIA Triad

* [[CIA Triad]]
    * Confidentiality Integrity Availability
    * Guiding model for designing information security policies.
    * Confidentiality
        * Keeping things hidden - using password protection.
    * Integrity
        * Making sure that the files you request are not changed in transit.
    * Availability
        * Information you have is readily accessible to those that should have it.

### Essential Security Terms

* Risk
    * Possibility of suffering a loss in the event of attack.

* Vulnerability
    * A flaw in the system that can be comprimised to exploit the system.

* [[Zero Day]]
    * 0-day vulnerability
    * Vulnerability not known to software vendor, but known to attacker.
    * Name refers to how long vendor has had to fix the issue.

* [[Exploit]]
    * Software used to take advantage of a vulnerability.

* Threat
    * Possibilities of an attack.

* Hacker
    * Someone that attempts to exploit a system.
    * Can be [[Black Hat]] or [[White Hat]].

* Attack
    * An attempt at causing harm to system.

### Malicious Software

* [[Malware]]
    * Malicious software.
    * Viruses, Worms, Adware, Trojan, Spyware (key loggers) etc.

### Antimalware Protection

* Malware removal
    * Gather and verify
        * Find out what the user did.
        * Understand the change of behaviour of their system.
    * Quarantine malwayre
        * Disconnect internet.
        * Disable automatic system backup.
    * Remove malware
        * Run offline malware scan.
        * Once done, create an automatic backup point.
    * Malware education.

### Malware Continued

* [[Botnets]]
    * Exploiting a system in order to utilise its resources for things like mining Bitcoin.

* Backdoors
    * A way that allows an attacker to gain access to your system.

* Rootkit
    * A collection of software that an admin would use to allow them to modify the operating system.

### Network Attacks

* [[DNS Cache Poisoning]]
    * An attack that works by tricking a DNS server into accepting a fake DNS record that points to a comprimised server.

* [[Man In The Middle Attack]]
    * Places the attacker in the middle of two hosts that think they're communicating directly with each other.
    * Session/cookie hijacking.

* [[Rouge AP]]
    * Example of [[Man In The Middle Attack]].
    * An access point that is installed on network without network administrators knowing.

* [[Evil Twin]]
    * Create a network that is identical to another to fool users.

### Denial-of-Service

* [[Denial-of-Service Attack]]
    * Aka DOS attack.
    * Take up resources of a service, to prevent real user's from access it.

* [[Ping Flood]]
    * Send many [[ICMP]] [[Echo Request]]s.

* [[Syn Flood]]
    * Send many [[SYN]] packets to servers.
    * Forcing server to send SYN-ACK, the attacker never responds with an ACK message.
    * Since it forces the connection to remain open, it's also known as [[Half-Open Attack]].

* [[Distributed Denial-of-Service Attack]]
    * Aka DDOS.
    * Like a [[Denial-of-Service Attack]], but using multiple computers.

### Other Attacks

* [[Injection Attacks]]
    * When an attack injects malicious code.
    * [[Cross-site Scripting]] (XSS) attacks.
        * Add some fake client-side code to misdirect your users.
    * [[SQL Injection]]
        * Allows an attack to send SQL commands to the database.

* [[Password Attacks]]
    * Utilising software to try to guess a user's pasword.
    * [[Brute-Force Attack]]
        * Keep trying different password combinations until your gain access.
        * Captures are used to differentiate between bots and users.
        * Strong passwords mitigate it by increasing the time to brute-force.

* [[Social Engineering]]
    * Relies on interactions with people: impersonating admins etc.

* [[Phising Attack]]
    * Trying to fool users by pretending to be an organisation they trust.
    * [[Spear Fishing Attack]], [[Phising Attack]] that targets an individual.
    * [[Whaling]] like spear fishing, but targetting high-level employee.

* [[Email Spoofing]]
    * Pretending to send email from a trusted email to exploit a user.

* [[Baiting]]
    * Entice a victim to do something: ie leaving a USB drive around to try to get the user to plug it in.

* [[Tail-Gating]]
    * Try to gain access to restricted area by following real employee in.

## Week 2

### Symmetric Encryption

* [[Encryption]]
    * Act of taking a message, called [[Plaintext]], and applying a [[Cipher]] to it to make it unreadable.
        * The unreadable text is called [[Ciphertext]].

* [[Decryption]]
    * Act of taking [[Ciphertext]] and converting into plaintext.

* Encryption algorithm
    * Underlying logic or process to convert [[Plaintext]] into [[Ciphertext]].
    * Usually complex mathematically operations.
    * Key is how algorithm is encrypted.

* [[Kerchkhoff's Principle]]
    * A cryptographic service should remain secure even if everything about system is known except for key.
    * Similar to [[Shannon's Maxim]]

* [[Cryptography]]
    * Discipline that covers practice of coding and hiding messages.
    * Study is referred to as cryptology.

* [[Cryptanalysis]]
    * Opposite of [[Cryptography]].
    * Process of deciphing coded messages.

* [[Frequency Analysis]]
    * Practice of looking at frequency of which letters appear in ciphertext.
    * Some types of encryption are vulnerable to this.

* [[Cryptanalysis Attacks]]
    * Most common types of attacks:
        * [[Known-Plaintext Analysis]]
            * Requires access to some of the plaintext to try to determine key.
        * [[Chosen-Plaintext Analysis]]
            * Attacker needs to know the encrpytion algorithm or have access to device.
        * [[Ciphertext-Only Analysis]]
            * Only requires access to encrypted messages.
            * A challenge that intelligence agencies face.
        * [[Adaptive Chosen-Plaintext Attack]]
            * Similar to [[Chosen-Plaintext Analysis]] except uses smaller plaintext samples.
        * [[Meddler-in-the-Middle]]
            * Insert a meddler between 2 devices.
            * Meddler replies as the user and performs a key exchange with each party.
    * Types of results from attack:
        * [[Instance deduction]]
            * Attacker discovers more plain or cipher text.
        * [[Information deduction]]
            * Attacker gets more info about text not previously known.
        * [[Distinguishing algorithm]]
            * Attacker finds encryption algorithm.
        * [[Global Deduction]]
            * Attacker finds algorithm that is functionally equivalent.
        * [[Total Break]]
            * Attacker gains entire key.

* [[Symmetric Cryptography]]
    * A type of encryption that uses the same key to encrypt and decrypt.
    * Example:
        * Substitution Cipher to replace some characters with different ones.
            * If sending and receiver uses same sub-table, then can be reversed.
        * [[ROT13]]
            * Rotated alphabet 13 places.
    * Symmetric key ciphers can be placed into 2 categories which replace to how ciphers operate on text.
        * [[Block Cipher]]
            * Takes data in and places into a fixed size bucket or block of data.
            * Encodes entire block as a unit.
            * Pads if data isn't big enough for block size.
        * [[Stream Cipher]]
            * Takes stream of input and encrypts one stream at a time.
            * Typically faster than [[Block Cipher]] and less complex to implement.
            * Can be less secure if key handling not done right.
        * If same key is used to encrypt data 2 or more times, it's possible to break cipher.
            * Therefore, [[Initialization Vector]] (IV)
                * Some random data that is integrated into key.
                * Must be send in plain text along with encrypted message.
                * 802.11 frame of encrypted wireless packet is example of this.
* [[Data Encryption Standard (DES)]]
    * Designed in 1970s.
    * Form of [[Symmetric Cryptography]] [[Block Cipher]].
    * Uses 64-bit key sizes, and operates on blocks of size 64-bit.
    * Key size is technically 64-bit in length, 8-bits is used for parity checking, so real-world length is 56.

* [[Key Length]]
    * Defines maximum potential strength of system.
    * Longer keys protect against brute force attacks.
    * A length of 56 gives 2^56 keys, which is actually quite small and can be brute forced easily.

* [[Advanced Encryption Standard (AES)]]
    * Replacement for [[Data Encryption Standard (DES)]].
    * Another form of [[Symmetric Cryptography]] [[Block Cipher]].
    * Use 128-bit blocks.
    * Supports [[Key Length]] of 128-bit, 192-bit or 256-bit.
    * Brute force attacks are theoretic with current computing power.
    * Modern CPUs from Intel or AMD have AES instructions built into the CPUs to improve speed and efficiency.

* [[Rivest Cipher 4 (RC4)]]
    * Supports key sizes from 40 bits to 2048 bits.
    * Cipher has inherent weaknesses and vulnerabilities, that have been discovered by recent attacks.
    * Was used in [[WEP]] and [[WPA]], and [[TLS]] until 2015.
    * Modern browsers have dropped support for it.
    * TLS 1.2 uses AES GCM.
        * Effectively turns [[Advanced Encryption Standard (AES)]] into a [[Stream Cipher]].

* Consider speed and ease of implementation when picking an encryption algorithm.

### Public Key or Asymmetric Encryption

* [[Asymmetric Cryptography]]
    * Aka public key ciphers.
    * Different keys and used encrypt and decrypt (public and private keys).
    * To exchange messages with a recipient, you must first share public key.
    * Works well in untrusted environments, but is less efficient than [[Symmetric Cryptography]]
    * Many secure communication protocols, use both asymmetric and symmetric.
        * Use asymmetric encryption for key exchange.
        * Then, once both have the key, can use symmetric encryption.

* [[Public Key Signatures]]
    * Combine a message with your private key to create a digital signature.
    * Sender can confirm the message hasn't been modified.
* [[Message Authentication Codes]]
    * Bit of information that allows auth of received message to ensure it wasn't tampered with.
    * Differs from public key cryptography, as the secret keys used to generate the MAC is the same one that verifies it.
    * [[HMAC]]
        * Uses a cryptographic hash function along with a secret key to generate a MAC
    * [[CMAC]]
        * Cipher-Based Message Authentication Codes.
        * Uses symmetric cipher with a shared key is used to encrypt the message and the resulting output is used as the MAC
        * Similar to HMAC, but doesn't use hashing function for digest
    * [[CBC-MAC]]
        * Mechanise for building MACs using [[Block Cipher]].
        * Take message and encrypt using block cipher operating in CBC mode.

        * [[CBC Mode]]
            * Operating mode for [[Block Cipher]]s that incorporates previously encrypted blocks ciphertext into the next blocks of plain text, to build a chain of encrypted blocks that requires the full unmodified chain to decrpyt.
            * Any modification to the plaintext will result in different final output at the end.

#### Asymmetric Encryption Algorithms

* [[RSA]]
    * Defines mechanisms for **generating and distributing** keys.
    * One of the first practical [[Asymmetric Cryptography]] systems.
    * Involves choosing 2 unique, random and large prime numbers.
    * Named after the first initials of 3 co-inventors: Ron Rivest, Adi Shamir, and Leonard Alderman
    * Patented in 1983 and released to public domain in 2000.

* [[DSA]]
    * Digital signature algorithm
    * Used for signing and verifying data.
    * Patented in 1991.
    * Part of US Governments federal info processing standard.
    * Lies on choosing a random seed.
        * If seed is leaked or can be inferred, someone can recover the private key.

* [[Diffie-Hellman]]
    * A key exchange algorithm.
    * Aka DH.
    * How it works:
        * 2 parties agree on large random starting number, which is different for each session.
        * Then, each person chooses another random large number and keeps it secret.
        * Combine shared number with respective secret and send mixed number result to each other.
        * Next, each person combines their secret number with the combined value they received from the previous step.
        * The result is a new value that's the same on both sides without disclosing enough info for some to figure out shared secret.

* [[Elliptic Curve Cryptography]]
    * A public-key encryption system
    * Aka ECC.
    * Makes use of [[Elliptic Curves]] instead of large prime numbers.
        * Some useful properties of Elliptic Curves:
            * Horizontal symmetry.
            * Any non-vertical line will intersect curve in 3 places at most.
              ![[elliptic-curve-non-vertical-lines.png]]
                 * This is the property that allows it to be used in encryption.
     * Benefit of elliptic curve based encrpytion systems, is it achieves security similar to tranditional public key systems with smaller key sizes.
         * A 256 bit elliptic curve would be comparable to a 3072 bit RSA key.
     * [[Diffie-Hellman]] and [[DSA]] have elliptic curve variants:
         * [[ECDH]]
         * [[ECDSA]]
     * The US NIST recommends the use of EC encryption, and the NSA allows its use to protect up to top secret data with 384 bit EC keys.
     * Some concerns its vulnerable to quantum computing attacks.

### Hashing

#### Hashing

* [[Hash Function]]
    * Operation that takes in data input and maps to output of fixed size, called hash or digest.
    * Core idea is that 2 different inputs should never have some hash.
    * Used in [[Hash Tables]].
    * Can be used to identify duplicate data.
    * Must be deterministic: same input data should return same hash.
    * Should be one-way: can't convert from hash to plaintext.
    * Similar to [[Block Cipher]] in that they operate on the entire block of data.

#### Hashing Algorithms

* [[Hashing Algorithms]]
