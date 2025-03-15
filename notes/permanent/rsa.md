---
title: RSA
date: 2025-03-01 00:00
modified: 2025-03-01 00:00
summary: a public-key encryption system reliant on the practical difficulty of factorising large numbers
tags:
- Cryptography  
- ComputerSecurity  
---

**RSA** is a public-key encryption system based on the idea that two prime numbers are computationally easy to multiply together but very difficult to factor back into the original primes. Named after its three creators from MIT: Ronald Rivest, Adi Shamir, and Leonard Adleman.

In RSA and other public-key systems, there is a public key, which can encrypt messages and is freely available, and a private key, which can decrypt messages and is kept secret. We can use the keys to encrypt messages (using the recipient's public key so only they can decrypt) and digital signatures (using the sender's private key to prove authenticity).

However, RSA is a slow algorithm typically not used for directly encrypting user data; more often, it's used for transmitting shared keys for symmetric key cryptography (like AES), which is much faster for bulk encryption.

### Algorithm

To create a private and public key, we use the following steps:

1. Select two large prime numbers, $p$ and $q$. (In practice, each should be hundreds or thousands of bits long—commonly 2048 bits or more—to ensure security.)
2. Calculate $N = p \times q$ (the modulus).
3. Calculate [Euler's Totient Function](eulers-totient-function.md): $\phi(N) = (p-1)(q-1)$
4. Choose a public key $e$ that is relatively prime to $\phi(N)$. Two numbers are relatively prime when their greatest common divisor (GCD) is 1. (A common choice is $e = 65537$, a [Fermat Prime](fermat-prime.md). It has only two 1's in its binary representation, greatly reducing exponentiation time.)
5. Compute the private key $d$ as the modular multiplicative inverse of $e$ modulo $\phi(N)$. This means finding a value $d$ where: $(d \times e) \mod \phi(N) = 1$. This can be calculated using the [Extended Euclidean Algorithm](../../../permanent/extended-euclidean-algorithm.md).

$N$ and $e$ are **public key** components, represented as $(N, e)$.

$p$, $q$, and $d$ are the **private key** components, although $p$ and $q$ are typically discarded, and it's represented as $(N, d)$.

### Encryption and Decryption

Given $e$ and $N$, we can encrypt messages: $c = m^e \mod N$.
Given $d$, we can decrypt messages: $m = c^d \mod N$.

### Example

For a simple example (using small numbers for clarity):

* Choose primes $p = 61$ and $q = 53$
* Calculate $N = 61 \times 53 = 3233$
* Calculate $\phi(N) = (61-1) \times (53-1) = 60 \times 52 = 3120$
* Choose $e = 17$ (relatively prime to 3120)
* Calculate $d = 2753$ (since $17 \times 2753 \mod 3120 = 1$)

Public key: $(N=3233, e=17)$

Private key: $(N=3233, d=2753)$

To encrypt message $m = 123$: $c = 123^{17} \mod 3233 = 855$

To decrypt ciphertext $c = 855$: $m = 855^{2753} \mod 3233 = 123$

### Security Considerations

If the value of $N$ is not large enough, an attacker could factorise it, effectively allowing them to solve for $d$. Key sizes of at least 2048 bits are recommended for security through 2030.

Additionally, modern RSA implementations use secure padding schemes (like PKCS\#1 OAEP) to protect against various attacks, including chosen-ciphertext attacks.

Side-channel attacks can leak information about private keys during implementation, requiring additional countermeasures.

Lastly, RSA's security relies on the computational difficulty of the factoring problem, but quantum computers running Shor's algorithm could break RSA encryption, driving research into post-quantum cryptography alternatives.
