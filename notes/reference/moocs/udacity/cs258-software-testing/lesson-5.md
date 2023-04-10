---
title: "Lesson 5: Testing in Practice"
date: 2014-12-14 00:00
category: reference/moocs
parent: uol-discrete-mathematics
status: draft
---

# Lesson 5: Testing in Practice

* Overwhelmed By Success
  * Too many test reports; developers start to ignore them.
    * Report one bug, not all
* Bug triage
  * Severity of bugs are determined
  * Figure out which bugs are duplicates
  * Single defect could cause multiple symptoms
  * Multiple defects could cause single symptoms
* Difficulties with Bug Triage
  * Disambiguation
  * Core dump or stack trace
  * Search over version history
    * git-bisect
* Test Case Reduction
  1. Manual reduction
    * Eliminate part of the input and see if it triggers the failure
    * One you have a small test case, report that to developers
  2. Delta debugging
    * Automate process of finding bugs
