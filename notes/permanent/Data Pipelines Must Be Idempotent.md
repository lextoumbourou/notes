---
title: Data Pipelines Must Be Idempotent 
date: 2022-07-09 00:00
tags:
  - DataEngineering
cover: /_media/idempotence.png
status: draft
---

One of the most costly mistakes we can make when designing a data pipeline is failing to account for idempotence.

What do I mean by idempotence? If we run a pipeline multiple times, it will generate the same dataset.

Let me describe a few common scenarios: loading a dataset of raw XML records into a relational database for easy querying.

A nieve approach is to write a quick script that converts the XML into a set of `INSERT` SQL queries. What happens if the job dies halfway through? If you try to rerun it, you will now have duplicate records. Maybe you can clear the data first. But what happens if other systems depend on that data? Now you need to do a massive coordination job when you should have just made the pipeline idempotent in the first place.

Instead, take some time to determine what makes the source data unique, then replace the `INSERT` with `UPSERT` style queries.

Perhaps it already comes with a unique identifier, like a primary key from the source database. Great! Just use that.

Maybe you need to combine multiple bits of metadata to create a primary key. Or perhaps the entire contents of the record is the unique identifier. Maybe you need to hash the whole payload to represent uniqueness.

Idempotence also allows you to correct errors with your transformations without having to start again with the data loading process completely.

It's not just loading data pipelines that need to be idempotent. Transformations that run across datasets should also have idempotence as a first-class concern.

To do: finish this example.