---
title: Idempotent Data Pipelines
date: 2022-07-10 00:00
tags:
  - DataEngineering
cover: /_media/idempotence.png
status: draft
---

One of the best bits of advice I'd give someone operating a data pipeline from simple to extreme complexity is to plan for idempotence.

If a function is idempotent, it means we can run it multiple times against input, and it will only transform it once:

```
f(f(x)) = f(x)
```

For a data pipeline that loads source data, transforms it, and stores it in a database, if we intend to generate N records, we should only have N records if we rerun it multiple times.

Pipelines will eventually, for reasons within and outside your control.

If each failure requires manual intervention to clear and rerun the entire pipeline from scratch, you may soon find all your time consumed maintaining the pipeline.

Instead, we want to fix the issue and rerun the last step.

In practice, this means using an `UPSERT` style query (or `DELETE` then `INSERT` if there are no other options) instead of `INSERT`.

But for this, we will need to think hard about how to identify the inputs and outputs of our data pipeline uniquely.

Sometimes we thought the source data with a unique primary key. But when we try to make our pipeline idempotent, we learn that what we thought was a unique record identifier was not so.

At times we will need to concatenate metadata to identify records.

Sometimes the entire contents of the source record are the identifier.

If you don't have idempotence now, it's time to think about how you can get it.

Unfortunately, this is a lesson many learn the hard way.