---
title: MapReduce
date: 2025-08-14 00:00
modified: 2026-09-26 08:55
status: draft
---

**MapReduce** is a two-phase distributed processing model described in paper [MapReduce: Simplified Data Processing on Large Clusters](mapreduce-simplified-data-processing-on-large-clusters.md) for processing large datasets, where the first "map" phase is used to filter and transform data locally where it resides, then a "reduce" phase aggregates and summarises the results.

The approach aims to minimise expensive data movement across networks while providing scalability and fault tolerance for big data processing tasks like word counting, where individual words are extracted and counted locally before being aggregated into final frequency statistics.