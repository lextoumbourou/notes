# Chapter 1. Meet Hadoop

* Disks speeds aren't keeping up with transfer speeds and reading and writing to disks can be slow.
  * Solution: Use multiple disks and read a section of the data from each.
    * First problem: Hardware failure risk increases
      * Use replication, like how RAID works.
    * Second problem: Combining data collected from disks.
      * MapReduce.
    * Hadoop provides all this functionality.
* MapReduce is fudamentally a batch processing system
  * Not suitable for interactive analysis: take that shits offline.
* Hadoop has evolved beyond batch processing
  * HBase
    * k/v store with random access
  * YARN (Yet Another Resource Negotiator)
    * cluster resource management system
    * has allowed new processing patterns:
      * Interactive SQL
        * Always on dedicated daemons (Impala)
        * Container reuse (Hive)
      * Iterative procesing
        * Spark
        * MapReduce doesn't work well with iterative algorithms that need to hold intermediate working set in memory
      * Stream processing
        * Run distributed computation on streams of data
        * Storm, Spark Streaming
      * Search
        * Solr on Hadoop

## Comparision with Other Systems

* RDMS
  * Better for lots of updates, not large scale analysis across total dataset.
  * Seek time is improving more slowly than transfer
  * MapReduce streams through data and operates on the transfer, RDMS, using B-Tree is limited by seek time.
  * RDMS = more structured, MapReduce = schemaless or semi-structured data.
* Grid computing
  * Distribute work across cluster of machines using shared storage.
  * Good for compute-intense stuff but bad when a lot of data access is required.
  * Hadoop co-locates data with compute nodes and is good at conserving bandwidth.
