---
title: Easy Approach to Requirements Syntax (EARS)
date: 2024-02-15 00:00
modified: 2024-02-15 00:00
status: draft
---

**Easy Approach to Requirements Syntax** or **EARS** is a streamlined approach to writing requirements structured around five patterns.

**Ubiquitous (always occurring)**: Defines fundamental properties of the system.

*`"The <system name> shall <system response>"`*
    
**Event-driven**: Initiates when and only when in case of a trigger.

`"WHEN <trigger> <optional precondition> the <system name> shall <system response>"`
   
 **Unwanted behaviours**: Managing undesired occurrences such as errors, failures, faults, disturbances, and other unwanted behaviours.
 
 `"IF <unwanted condition or event>, THEN the <system name> shall <system response>"`
    
**State-driven**: Triggered while in a state.

`WHILE <system state>, the <system name> shall <system response>"`
    
**Optional features**: These are called upon exclusively within systems with specific optional features.

`WHERE <feature is included>, the <system name> shall <system response>`

It can also be a complex combination of those patterns.