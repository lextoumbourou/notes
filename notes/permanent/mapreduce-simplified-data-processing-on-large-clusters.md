---
title: "MapReduce: Simplified Data Processing on Large Clusters"
date: 2025-08-14 00:00
modified: 2026-09-26 08:55
status: draft
---

**MapReduce: Simplified Data Processing on Large Clusters** is a 2004 paper by Jeffrey Dean and Sanjay Ghemawat from Google, presented at OSDI.

It describes MapReduce, a programming model for processing large datasets across a cluster of machines. Users write two functions:

* **map**: takes an input key/value pair and produces a set of intermediate key/value pairs.
* **reduce**: merges all the intermediate values that share the same key.

The framework handles the hard parts: partitioning the input, scheduling work across machines, and recovering from machine failures. The classic example is counting how many times each word appears across a large collection of documents.

It inspired open-source implementations like Hadoop.
