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
