# Intro To Hbase Snapshots

* Each Hbase table is basically a set of metadata in key/value pairs:

  * **Table Info** - manifest files that describes "settings": column families, compression, encoding, bloom filter types etc.
  * **Regions** - table "partitions". Defines start and end keys for data.
  * **WALs/MemStore** - before writing to disk, PUTs are chucked in Write Ahead Log (WAL) then stored in-memory until shit needs to be persisted to disk.
  * **HFiles** - Hbase format that actually stores the data on disk (immutable - they are never changed, just archived and deleted).

* Snapshot is basically a reference to those files, which can be used to recover a table to a point in time.
* When files are referenced by snapshots, they take up space on disk. Once snapshots are cleared, they can be moved to archive and eventually deleted.

## References

* http://blog.cloudera.com/blog/2013/03/introduction-to-apache-hbase-snapshots/
* http://blog.cloudera.com/blog/2013/06/introduction-to-apache-hbase-snapshots-part-2-deeper-dive/
