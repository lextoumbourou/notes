---
title: Module Coupling
date: 2023-12-31 00:00
modified: 2023-12-31 00:00
summary: a measure of the interdependence between software modules
cover: /_media/module-coupling.png
tags:
  - SoftwareEngineering
---

Module coupling describes the idea of interdependence between software modules. How much do modules rely on each other?

Useful software, beyond some complexity, cannot exist without module coupling. However, some forms of module coupling are desirable, and others are not.

Like [Module Cohesion](module-cohesion.md), the [ISO/IEEE Systems and Software Engineering Vocabulary](https://www.iso.org/obp/ui/#iso:std:iso-iec-ieee:24765:en) recognises several subtypes of module cohesion.

## [Common Environment Coupling](common-environment-coupling.md)

**Common Environment Coupling** occurs when multiple software modules share the same global state or environment, for example, global variables, singleton state objects, system environment variables, etc.

It is not necessarily a bad thing. However, it can lead to difficulty finding bugs as different modules mutate the global state haphazardly.

Consider using a sub-environment, i.e. global states within specific classes or modules, instead of having all modules share an environment. If you must use a global environment, ideally ensure it's read-only.

## [Content Coupling](content-coupling.md)

**Content Coupling** is when a module is contained within another module. One example of this type of coupling is an Image module that contains various subimage implementations: `JpegImage`, `PngImage`, `GifImage`, etc. The user doesn't need to understand which submodule to call for their specific image type; they request to load an image, and the main module calls the required submodules.

Content Coupling is universally considered a good idea and property of Khorikov's [Well-Designed API](well-designed-api.md).

## [Control Coupling](control-coupling.md)

**Control Coupling** is when a module communicates information to another to influence its execution—for example, passing flags from `ModuleA` to `ModuleB` to change a mathematical operation that `ModuleB` performs.

Control coupling is mostly bad; in the example above, the issue is that `ModuleB` is hard to test and verify since it's dependent on control information from `ModuleA`, and the design is complex and hard to reason about.

## [Data Coupling](data-coupling.md)

**Data Coupling**, also known as input-output coupling, is a type of coupling in which output from one software module is input to another.

Example:

```python
data = prepare_data()
process_data(data)
```

Data coupling is mostly good—a better alternative than [Control Coupling](control-coupling.md).

## [Hybrid Coupling](hybrid-coupling.md)

**Hybrid Coupling** occurs when different subsets of the range of values of a data item are used for separate and unrelated purposes.

It's generally considered a bad thing, although sometimes it's the only option, especially in limited memory environments (microcontrollers).

## [Pathological Coupling](pathological-coupling.md)

**Pathological Coupling** occurs when one module completely changes the behaviour of another module. For example, monkey patching or modifying private variables to change the behaviour.

Unsurprisingly, it's universally considered a bad thing. Unless you have a very good reason to do it, consider refactoring.
