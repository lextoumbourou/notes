---
title: Module Cohesion
date: 2023-12-29 00:00
modified: 2023-12-29 10:00
summary: How elements of a module relate to each other
cover: /_media/module-cohesion.png
tags:
  - SoftwareEngineering
---

In software engineering, module cohesion refers to the degree to which the elements of a [Module](module.md) are related. For example, functions that manage the same type of data or contribute to a singular, unified piece of capability.

Some forms of cohesion are considered good, while others are not.

Like [Module Coupling](module-coupling.md), the [ISO/IEEE Systems and Software Engineering Vocabulary](https://www.iso.org/obp/ui/#iso:std:iso-iec-ieee:24765:en) recognises seven types of module cohesion.

## [Functional Cohesion](functional-cohesion.md)

Functional cohesion refers to when elements of a module are grouped because they contribute to the same purpose.

For example, a module called `file_operations` has a set of functions like `read_file()`, `write_file()` and `delete_file()`. Each function contributes to the overarching file-handling capability of the program.

Functional cohesion is the ideal type and is universally considered good.

## [Communicational Cohesion](communicational-cohesion.md)

Communicational cohesion occurs when code is grouped because it operates on the same data type.

For example, an `Invoice` class might contain methods for adding items, calculating the total, applying discounts, etc. Each method operates on the same data structure, the items in an invoice.

As the foundation of object-oriented programming, communicational cohesion is generally considered good. However, consider a complex `Customer` class, where we have methods for adding customers, updating details, generating customer-specific reports, etc. Though the module operates on the same data type, the functionality is quite different, and refactoring it into multiple modules might be better.

## [Logical Cohesion](logical-cohesion.md)

Logical cohesion is when things are grouped because they relate in some way but don't necessarily have interdependence or a relationship to a single task.

For example, a math module includes functions like `math.square_root`, `math.log`, `math.sine`.

Sometimes considered a bad practice - while they may be related logically, they don't share a relationship that warrants putting them in a single module.

## [Procedural Cohesion](procedural-cohesion.md)

Procedural cohesion is when things are grouped because they happen in sequence.

For example, a process class contains a method for user authentication, followed by a technique to log the authentication, and finally, a method to redirect the user based on their role. These methods are part of a sequence (login process) but are quite different in functionality.

Generally considered bad. Just because things happen sequentially doesn't mean they should be logically grouped.

## [Sequential Cohesion](sequential-cohesion.md)

Sequential cohesion is where the output of one task performed by a software module serves as input to another task performed by the module.

For example, in a data processing module, one function takes raw data and formats it, the following function performs some calculations on this formatted data, and a third function logs the results. Here, the output of one function becomes the input for the next.

Similar to procedural cohesion, it's considered bad.

## [Temporal Cohesion](temporal-cohesion.md)

Temporal cohesion refers to grouping based on things that happen at a similar time or a particular program execution phase.

An example is a startup module that initialises various components of an application - like setting up database connections, loading configuration files, and initialising logging. These tasks are all required during the startup phase but aren't functionally related, indicating temporal cohesion.

Mostly considered bad: just because things happen simultaneously doesn't mean they should be related.

## [Coincidental Cohesion](coincidental-cohesion.md)

Coincidental cohesion is where elements of a module share no functional relationship.

For example, a module contains miscellaneous functions like `calculate_age`, `generate_random_number`, `format_date` etc. Here, these tasks have no functional relationship to each other and are grouped by coincidence, not because they logically belong together.

Coincidental cohesion is universally considered bad.
