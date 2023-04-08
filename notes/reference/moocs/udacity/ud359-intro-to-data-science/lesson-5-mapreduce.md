---
title: Lesson 5: Map Reduce
date: 2021-10-30 00:00
modified: 2023-04-08 00:00
---
# Lesson 5: MapReduce

* Only suitable for reaaaaally big data
    * All the books ever written
* Process
    1. Mapper emits key-value pairs to standard out
    2. Data is shuffled and sent to reducer
    3. Reducer processes the data
* Map reduce ecosystem
    * Hive
        * Run map-reduce through SQL-like language
    * Pig
        * Write queries in a procedural language
        * "Split your data pipeline" (what?)
        * Can do joins between datasets
