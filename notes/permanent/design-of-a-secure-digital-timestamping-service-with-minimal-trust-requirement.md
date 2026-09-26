---
title: Design of a Secure Digital Timestamping Service With Minimal Trust Requirement
date: 2025-03-15 00:00
modified: 2026-09-26 08:50
status: draft
---

*Notes from paper [Design of a Secure Digital Timestamping Service With Minimal Trust Requirement](https://cdn.nakamotoinstitute.org/docs/secure-timestamping-service.pdf) by H. Massias, X. Serret Avila, J.-J. Quisquater*

---

This paper presents a design for a secure timestamping service, focused on creating digital timestamps with minimal trust requirements.

Aspects of the system include:

**Timestamping Approach** - authors chose a [Binary Tree](binary-tree.md) structure that works in rounds of fixed duration. Each timestamp request (document hash) becomes a leaf in the tree, and hashes are combined pairwise up to a single root hash value. Uses two hash algorithms, SHA-1 and RIPEMD-160 for redundancy.

**Trust Model** - instead of requiring complete trust in a third party (Secure Timestamp Authority or STA) the system published certain "Big Round Values" to unmodifiable, widely witnessed media (like newspapers). This provides a trust anchor that all verifiers can reference.

**System Architecture**

The implementation uses a multi-threaded approach with several components:

- Network Listener: Receives timestamp requests
- Request Timer: Orders and timestamps incoming requests
- Round Queue Coordinator: Manages requests for each round
- Timestamp Generator: Creates timestamps from the binary trees
- Network Answer: Forwards timestamps to clients

**Key Processes**:

* Document timestamping: Documents are hashed, processed in binary trees, and linked to previous rounds
* Timestamp verification: Verifiers can rebuild tree branches to confirm validity
* Auditing: The system supports auditing to check consistency between published Big Round Values
* System start-up/shutdown: Special procedures ensure continuity even after unexpected shutdowns

**Security Features**:

* The binary tree approach prevents the STA from creating undetectable backdated timestamps
* Using two hash functions protects against the unexpected failure of one algorithm
* Periodic publication of Big Round Values creates widely witnessed trust anchors
* Support for re-timestamping to extend lifetime beyond cryptographic signature lifetimes