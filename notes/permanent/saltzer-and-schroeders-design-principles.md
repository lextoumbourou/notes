---
title: Saltzer and Schroeder's Design Principles
date: 2024-02-15 00:00
modified: 2024-02-15 00:00
status: draft
---

The design principles established by Saltzer and Schroeder in their 1975 work, "The Protection of Information in Computer Systems," were intended to guide the creation of secure systems. These principles are still influential in the design of secure software systems. Here's a summary of the principles:

**Economy of mechanism**: This principle emphasizes that the system's design should be as simple and small as possible, which aids in managing security and understanding the system.

**Fail-safe defaults**: The default stance of a system should be secure, with access decisions based on explicit permission rather than an exclusion list; this means that unless a user is given explicit access, they should not have it.

**Complete mediation**: This principle requires that every access to every resource must be checked for proper authorization, ensuring no gaps in the protection.

**Open design**: The security of a system should not rely on its design's secrecy but rather on its implementation's security, meaning that the mechanisms should be open to scrutiny.

**Separation of privilege**: A system should not grant permission based on a single condition. Instead, it should require multiple conditions to be met, increasing security through redundancy.

**Least privilege**: Users and programs should operate using the least privilege necessary to complete their tasks, limiting the potential damage from accidents or attacks.

**Least common mechanism**: The system should minimize the number of shared mechanisms to reduce the chances of a breach that could affect all users.

**Psychological acceptability**: Security mechanisms should not make the system unwieldy or hard to use, as this can lead to insecure practices by users.

**Work factor**: The cost of circumventing a security mechanism should be compared with the resources and capabilities of a potential attacker, ensuring that it is not worth the effort to breach the system.

**Compromise recording**: While not always suggested as a replacement for more robust security measures, having a reliable way to record compromises can be used as an additional layer of security.

These principles form a foundation for thinking about system security from a comprehensive and strategic standpoint, ensuring that the security mechanisms are integrated thoughtfully into the overall system design.
