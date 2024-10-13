---
title: Easy Approach to Requirements Syntax (EARS)
date: 2024-02-15 00:00
modified: 2024-02-15 00:00
status: draft
---

**Easy Approach to Requirements Syntax** or **EARS** is a streamlined approach to writing requirements structured around five patterns, described in paper [[EARS (Easy Approach to Requirements Syntax]]

Think of the generic requirements system. It would usually consistent of something like this:

```
<Optional preconditions>
<Optional trigger> the <system name> shall <system response>
```

Example of a user clicking on a profile:

```
Pre: The user is logged in
Trigger: They click on the profile button
System: The user profile drop down.
Response: Appears on the top left of the screen
```

EARS broke down into 5 main ideas:

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
