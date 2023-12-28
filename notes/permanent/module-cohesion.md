---
title: Module Cohesion
date: 2023-12-29 00:00
modified: 2023-12-29 00:00
summary: how elements of a module relate to each other
cover: /_media/module-cohesion.png
tags:
  - SoftwareEngineering
---

Module Cohesion refers to how elements of the module of code relate. For example, some functions may operate on the same data type; others perform functionality contributing to one core capability.

Some forms of cohesion are desirable, while others are not.

The [ISO/IEEE Systems and Software Engineering Vocabulary](https://www.iso.org/obp/ui/#iso:std:iso-iec-ieee:24765:en) describes seven types of module cohesion.

## [Functional Cohesion](functional-cohesion.md)

Functional cohesion refers to when elements of a module are grouped because they contribute to the same purpose.

For example, a module called `file_operations` has a set of functions like `read_file`, `write_file` and `delete_file`. Each function contributes to the overarching file-handling function - each task contributes to a well-defined thing.

Functional cohesion is the ideal type and is almost always considered good.

## [Communicational Cohesion](communicational-cohesion.md) 

Communicational cohesion occurs when code is grouped because it operates on the same data type.

For example, an `Invoice` class might contain methods for adding items, calculating the total, applying discounts, etc. Each method operates on the same data structure, the items in an invoice.

As the foundation of object-oriented programming, communicational cohesion is generally considered good.

## [Logical Cohesion](logical-cohesion.md)

Logical cohesion is when things are grouped because they relate in some way but don't necessarily have interdependence or a relationship to a single task.

For example, a math module includes functions like `math.square_root`, `math.log`, `math.sine`.

Generally considered bad practice - while they may be related logically, they don't share a relationship that warrants putting them in a single module.

## [Procedural Cohesion](procedural-cohesion.md)

Procedural cohesion is when things are grouped because they happen in sequence.

For example, a process class contains a method for user authentication, followed by a technique to log the authentication, and finally, a method to redirect the user based on their role. These methods are part of a sequence (login process) but are quite different in functionality, illustrating procedural cohesion.

Generally considered bad. Just because things happen sequentially doesn't mean they should be logically grouped.

## [Sequential Cohesion](sequential-cohesion.md)

Type of Cohesion in which the output of one task performed by a software module serves as input to another task performed by the module.

For example, in a data processing module, one function takes raw data and formats it, the following function performs some calculations on this formatted data, and a third function logs the results. Here, the output of one function becomes the input for the next, demonstrating sequential cohesion.

Similar to procedural cohesion, it's considered bad.

## [Temporal Cohesion](temporal-cohesion.md)

Temporal cohesion refers to grouping based on things that happen at a similar time or a particular program execution phase.

An example is a startup module that initialises various components of an application - like setting up database connections, loading configuration files, and initialising logging. These tasks are all required during the startup phase but aren't functionally related, indicating temporal cohesion.

Sometimes considered bad: just because things happen simultaneously doesn't mean they should be related.

## [Coincidental Cohesion](coincidental-cohesion.md)

No functional relationship.

Type of Cohesion in which tasks performed by a software module have no functional relationship to one another.

Example: A utility module containing a random assortment of functions such as `calculate_age`, `generate_random_number`, and `format_date`. 

These tasks have no functional relationship to each other and are grouped by coincidence, not because they logically belong together, exemplifying coincidental cohesion.

Coincidental cohesion is universally considered bad.