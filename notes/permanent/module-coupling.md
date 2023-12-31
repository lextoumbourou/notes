---
title: Module Coupling
date: 2023-12-31 00:00
modified: 2023-12-31 00:00
summary: a measure of the interdependence between software modules
cover: /_media/module-coupling.png
---

Module coupling describes the idea of interdependence between software modules. How much do modules rely on each other?

Like [Module Cohesion](module-cohesion.md), the [ISO/IEEE Systems and Software Engineering Vocabulary](https://www.iso.org/obp/ui/#iso:std:iso-iec-ieee:24765:en) recognises a number of subtypes of module cohesion. 

## [Common Environment Coupling](common-environment-coupling.md)

Common Environment Coupling occurs when multiple software modules share the same global state or environmental variables. This coupling is often seen in using global variables or singleton state objects.

It is not necessarily a bad thing. However, it can lead to difficulty finding bugs as different modules mutate the global state haphazardly.

One way around it is to create a sub-environment, i.e. global states within specific classes or modules. Or still with read-only global environments.

## [Content Coupling](content-coupling.md)

Content Coupling is when we have a module contained within another module.

Consider this Python example, where the Image module contains various image implementations. The user doesn't need to understand all the modules' implementations.

```

class Image(object):
    def load_from_filename(self, filename):
        if filename.endswith("jpg"):
            return JpegLoad(filename)
        if filename.endswith("png"):
            return PngLoad(filename)
        raise InputError(f"Unsupported file {filename}")

class JpegLoad(object):
    # ...

class PngLoad(object):
    # ...

```

Pass event listeners are another example of content coupling, where one module is used in another but has a one-way relationship.

## [Control Coupling](control-coupling.md)

Control Coupling is when a module communicates information to another module to influence its execution. 

For example, you perform certain math operations if a flag is passed in.

Control coupling is mostly bad; it is better not to have a complex module but to split it into multiple modules that can be tested individually.

## [Data Coupling](data-coupling.md)

They are also known as input-output coupling.

Type of coupling in which output from one software module serves as input to another.

Example:

```python
data = prepare_data()
process_data(data)
```


Data coupling is mostly good. A better alternative than [Control Coupling](control-coupling.md).

## [Hybrid Coupling](hybrid-coupling.md)

Different subsets of the range of values of a data item are used for separate and unrelated purposes.

It's generally considered a bad thing, although sometimes it's the only option, especially in limited memory environments (microcontrollers).

## [Pathological Coupling](pathological-coupling.md)

When one module completely changes the behaviour of another module. For example, monkey patching or modifying private variables to change the behaviour.