[[Idempotence]] in a data pipeline ensures that running the same task multiple times will generate the same final output.

Usually this is achieved by ensuring each "record", or item processed, contains a unique identifier allowing for an [[upsert]] style of record insertion.

In my experience, data pipelines without idempotence are near impossible to operate at any significant scale of dataset size. 

A similar sentiment was shared in [Eric Lathrop's](https://ericlathrop.com/) blog post, [[Idempotence Now Prevents Pain Later]], which describes a scenario where converting a billing cron script to an idempotent script significantly reduces the overhead in running the script.

Idempotence alone in a data pipeline does not guarantee a pipeline will not be resumable, where if a job fails it can be rerun without reprocessing things that have already been processed.

---

Tags: #Data 
Reference: [[Idempotence In Data Pipelines]]