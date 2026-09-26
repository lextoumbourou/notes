---
title: Interned Strings
date: 2024-12-22 00:00
modified: 2026-09-26 09:05
status: draft
---

**Interned Strings** are strings where only one copy of each distinct value is stored in memory, and every occurrence of that value points to the same object.

Because equal strings share the same object, they can be compared by identity (a pointer comparison) instead of character by character, and duplicate strings don't use extra memory. It only works for immutable strings.

Many languages do it, e.g. Python (`sys.intern`, and automatically for some strings like identifiers), Java (`String.intern()` and string literals) and Lua (for short strings).
