# Architecture

* Strongly consistent read / write - not "eventually consistent" 
* Automatic sharding: tables are distributed as regions which are auto split and re-distributed as data grows. 
* Automatic RegionServer failover (kinda).
* Hadoop/HDFS integration: HDFS is it's primary datastore.
* MapReduce: used as a source and sink.
* Java Client API, Thrift API for non-Java clients.
* Block Cache and Bloom Filters: for high-volume query optimisations.
* Operational Management: web-pages for Ops insights as well as JMX metrics.

## Catalog Tables

* Catalog table is called ``hbase:meta`` and is a table like any other but isn't displayed in the shell.
* Location of ``hbase:meta`` table is now stored in Zookeeper.
* HBase meta format looks likes this:
  * Key: region key of format ``([table],[region start key],[region id])``
  * Values:
    * ``info:regioninfo`` - serialized HRegionInfo instance for the region.
      * ``tableName`` - name of the table.
      * ``startKey`` - startKey of the region.
      * ``regionId`` - timestamp when region is created.
      * ``replicaId`` - id starting from 0 to differentiate replicas of the same region range.
      * ``encodedNmae`` - MD5 encoded string of the region name.
    * ``info:server`` - server:port of RegionServer containing the region.
    * ``info:serverstartcode`` - start-time of RegionServer proess containing this region.

## Startup Sequencing

* Location of ``hbase:meta`` is looked up in ZooKeeper.
* ``hbase:meta`` is updated with server and startcode values.

## Client

* Client asks ZooKeeper for the ``hbase:meta`` table info.
* Goes to server with ``hbase:meta`` and finds the region of interest.
* Issues read and write requests directly to that region.
* If region is unavailable (has been moved or region server has died), client should requery catalog table for new region location.

## Client Request Filters

* There are many types of filters - best to understand the groups of filter functionality.

### Structural

* Contain other filters.

#### FilterList

* Represents a list of Filter with relationship either ``FilterList.Operator.MUST_PASS_ALL`` (``AND`` filter) or ``FilterList.Operator.MUST_PASS_ONE`` (``OR`` filter).

  * Example of an ``OR``-type filter (``MUST_PASS_ONE``):
   
    ```
    FilterList list = new FilterList(FilterList.Operator.MUST_PASS_ONE);
    SingleColumnValueFilter filter1 = new SingleColumnValueFilter(
      cf, column, CompareOp.EQUAL, Bytes.toBytes('nirvana')
    );
    list.add(filter1);

    SingleColumnValueFilter filter2 = new SingleColumnValueFilter(
        cf, column, CompareOp.EQUAL, Bytes.toBytes('foo fighters'));
    list.add(filter2);

    scan.setFilter(list);
    ```

### Column Value

* Obviously just used to filter on value of a single column.

#### SingleColumnValueFilter

* ``CompareOp.EQUAL`` - equivalence filter.

  ```
  SingleColumnValueFilter filter = new SingleColumnValueFilter(
    cf, column, CompareOp.EQUAL, Bytes.toBytes('australia'));
  scan.setFilter(filter);
  ```

* ``CompareOp.NOT_EQUAL`` - inequality filter.

  ```
  SingleColumnValueFilter filter = new SingleColumnValueFilter(
    cf, column, CompareOp.NOT_EQUAL, Bytes.toBytes('new zealand'));
  scan.setFilter(filter);
  ```

* ``CompareOp.GREATER`` - range filter.

  ```
  SingleColumnValueFilter filter = new SingleColumnValueFilter(
    cf, column, CompareOp.NOT_EQUAL, Bytes.toBytes('2016-06-15T22:02:59.327214'));
  scan.setFilter(filter);
  ```

### Column Value Comparators

#### RegexStringCompatator

* Let's you use regexes for value comparisons.

  ```
  RegexStringComparator comp = new RegexStringComparator('prefix.');
  SingleColumnValueFilter filter = new SingleColumnValueFilter(
    cf, column, CompareOp.EQUAL, comp);
  scan.setFilter(filter);
  ```

#### SubstringComparator

* Determine if substring exists in a value.

  ```
  SubstringComparator comp = new SubstringComparator('teen spirit');
  SingleColumnValueFilter filter = new SingleColumnValueFilter(
    cf, column, CompareOp.EQUAL, comp);
  scan.setFilter(filter);
  ```

#### BinaryPrefixComparator

* Compare with prefix of byte array.

#### BinaryComparator

* Compare bytes arrays.

### KeyValue Metadata

* Use for comparing exists of keys (``ColumnFamily:Column qualifiers``) for a row.

#### FamilyFilter

* Used to filter on ColumnFamily. Better to select ColumnFamilies in the scan.

#### QualifierFilter

* Filter based on Column (aka Qualifier) name.

#### ColumnPrefixFilter

* Use for prefix of columns. "Find all columns that start with ``yadda``".

#### MultipleColumnPrefixFilter

* Same as ``ColumnPrefixFilter`` but for specifying multiple prefixes.

#### ColumnRangeFilter

* Use for intra row scanning (when you have *really* wide rows).

### RowKey

#### RowFilter

* Similar to using ``startRow/stopRow`` methods on Scan.

### Utility

#### FirstKeyOnlyFilter

Primary for rowcount jobs.

## Master

* ``HMaster`` - implementation of Master Server.
* Usually run name node.
* With multi masters - all compete to run cluster. If active master loses lease in ZooKeeper, all remaining Master try to get role.
* Cluster can still survive without master, however, region failover and region splits won't work, so cluster will soon fail.

### Interface

* ``HMasterInterface`` methods:
  * Table-related: ``createTable``, ``modifyTable``, ``removeTable``, ``enable`` or ``disable``
  * ColumnFamily: ``addColumn``, ``modifyColumn`` or ``removeColumn``
  * Region (move, assign, unassign operations)

### Processes

* Runs the following background threads:
  * ``LoadBalancer`` - runs periodically to balance cluster's load.
  * ``CatalogJanitor`` - checks and cleans up ``hbase:meta`` table periodically.

## RegionServer

* ``HRegionServer`` - implementation of RegionServer.
* Runs on Hadoop DataNode.
* Serves regions.

### Interface

* Data (get, put, delete, next).
* Region (splitRegion, compactRegion etc)

### Processes

* ``CompactSplitThread`` - checks for splits and handle minor compactions.
* ``MajorCompactionChecker`` - check for major compactions
* ``MemStoreFlusher`` - flush in-memory writes to StoreFiles.
* ``LogRoller`` - check the RegionServer WAL.

## Coprocessors

* Code that runs on RegionServers in response to certain events (data manipulation, WAL-related ops etc).

## Block Cache

* ``LruBlockCache`` - default on-heap cache.
* ``BucketCache`` - usually off-heap.
  * Memory not managed by GC, so latencies is less erratic but potentially slower on average.
  * Creates a "two-tier" caching system:
    * L1 cache (LruBlockCache)
    * L2 cache (BucketCache
  * Combined by ``CombinedBlockCache``

### LruBlockCache Design

* LRU cache with 3 levels of block priority:
  * Single access priority: When block is first loaded from HDFS, it has this priority. Considered first for evictions.
  * Multi access priority: graduates to this from first group once it is accessed again.
  * In-memory access priority: used when a block is configured to be "in-memory" (catalog tables are configured like this).
    * Call ``HColumnDescriptor.setInMemory(true);`` to set this.
    * Or: ``create 't', {NAME => 'f', IN_MEMORY => 'true'}`` from the shell.

#### LruBockCache Usage

* Enabled by default for user tables.
* Important concept for future tuning block caching: "working set size" (WSS) - amount of memory needed to compute the answer to a problem
* Calculate memory required for HBase caching like: ``num region servers * heap size * hfile.block.cache.size * 0.99``
  * Default ``hfile.block.cache.size`` is 0.25 (25% of heap).
  * 0.99 = amount loaded in LRU before evictions happen.
* Other things that live in block cache:
  * Catalog Tables - ``hbase:meta`` tables have in-memory priority and are hard to evict.
    * Can take a few MBs depending on num regions.
  * HFiles Indexes.
    * File format HBase uses to store data in HDFS.
    * Has indexes to support fast seeks without reading whole file.
  * Keys.
    * Keys are stored along side data, so take into consideration.
  * Bloom Filters.
    * Also stored in the LRU.
* Bad to use block cache when WSS doesn't fit into memory.
  * If you have a fully random reading pattern: never access same row twice in a short period.
  * Map reduce jobs: each row will be read once, no need to put in block cache.
    * Question: how do I disable block caching via the Thrift API?
