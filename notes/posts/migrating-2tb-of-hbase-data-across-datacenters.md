---
date: 2023-04-16 00:00
---
Title: Migrating TB HBase datasets across datacenters
Date: 2016-06-19
Tags: HBase, Migration
Status: Draft

# Practical considerations for migrating TB datasets across datacenters

At Scruch, we recently switch hosting providers for Rackspace to AWS. One of the many migration tasks was to move our 2TB HBase cluster with very little to no downtime. This article is mostly a brain dump of the research that went into this migration and includes some practical advice about the migration that aren't really available in other articles.

## Requirements

The requirements for the migration were pretty straight forward:

* Copy 2TB (roughly x rows), across 10 tables from Rackspace's DLW datacentre to AWS North Virginia DC (us-west-1).
* Little, to no downtime.
* No data loss (obviously).

## Options considered

There are a few options for copying data across HBase clusters. [This Cloudera article](http://blog.cloudera.com/blog/2013/11/approaches-to-backup-and-disaster-recovery-in-hbase/), though purported to be about backup and disaster, lists most of choices. I had considered most of them and eventually settled on CopyTable.

### Export Snapshots

Definitely seems like a solid approach. However, at the time of writing, snapshots have a huge limitation in that there's no incremental capability. Since copying the entire dataset with a transfer rate of, say, 100MB/s (round about the best I could get in my tests) that would take about 11 hours to complete the entire dataset. Since we're constantly writing and changing that dataset, it wasn't feasible to have an outage that long.

However, if it is feasible for you to have an outage that long, this is probably the approach I'd take. You need to ensure that your source cluster can communicate with the remote Hadoop cluster, though if that's not possible, you could also consider shipping the tables to S3 as an intermidiate step.

### Replication

Replication is another solid migration choice and potentially a solid contender for the "best (tm)" choice. The theory here is you have a source cluster and a destination cluster and any activity on the is replicated across to the destination. When it's time to cut over, you just cut over and you good. However, replication only acts on new edits/writes after replication is turned on. You still need to copy up the data somehow. However, a combination of export snapshots and replication is probably a solid choice for most workloads.

### CopyTable

CopyTable is a util that ships with HBase post 9.x and runs as a MapReduce job against the source cluster doing bulk mutations against the remote cluster. In other words, it's a script that runs on our source cluster sending PUTs up rows to the remote cluster.

CopyTable accepts a ``starttime`` param which allows you to just copy up anything that has changed since the time in milliseconds. Note, however, that this does not include deletion events.

Anyway, CopyTable seems to fit the task requirements. So let's talk about it in practise.

## Setup

Okay, so now we know we're going to use CopyTable, let's get setup for using it over a internet.

### Ensure task nodes are configured correctly

Since CopyTable is simply a MapReduce job, you are required to run an instance of resource manager and an instance of a node manager. These can be started on a node as follows:

```
/bin/yarn resourcemanager start
```

```
/bin/yarn nodemanager start
```

The copyTable operation is reasonable heavy on memory, so it's probably not advised to colocate these daemons on a regionserver or the master server.

At Scrunch, we aren't really running MapReduce jobs yet, so there was a little bit of setup involved to prepare YARN. If you're familar with YARN, then you can safely skip these steps:

To do: list the steps here.

### Ensure source cluster can communicate with destination cluster

Since CopyTable performs standard puts on the destination cluster, the MR task machines must be able to access both ZooKeeper and the RegionServers. If you aren't familiar with how the HBase client works, it's something like this:

1. Ask ZooKeeper for RegionServer that has catalog region (``hbase:meta``).
2. Go to that RegionServer and query the Catalog for the region you want.

So, with that in mind you'll need to ensure that you can resolve the ZooKeeper hostname and also access it on it's port (by default 2181).

Since we are sending the data to a remote cluster, I mapped the internet IP address to the internal address from my remote cluster in the hosts file on each node:

```
> vi /etc/hosts
zookeeper.internal 50.23.23.12
region1.internal 50.23.23.12
region2.internal 50.23.23.12
region3.internal 50.23.23.12
```

Then setup my security groups (firewall rules in AWS-land) such that the source cluster could access the remote cluster on the following ports:

* ZooKeeper - 2181
* RegionServer RPC - ??

Here's a screenshot of the security group configuration for the destination cluster:

To do: add me.

## Preparing the destination cluster

This article assumes you have already configured HBase on your destination cluster.

### Ensure tables exist on destination with same column-families as source

CopyTable requires that the destination cluster has the same column-family configuration as the source. You can see how your source in configured using ``describe`` from the shell.

```
hbase(main):005:0> describe 'my_table'
Table my_table is ENABLED
my_table
COLUMN FAMILIES DESCRIPTION
{NAME => 'a', BLOOMFILTER => 'NONE', VERSIONS => '3', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'FALSE', D
ATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'f
alse', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
{NAME => 'i', BLOOMFILTER => 'NONE', VERSIONS => '3', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'FALSE', D
ATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'f
alse', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
{NAME => 'r', BLOOMFILTER => 'NONE', VERSIONS => '3', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'FALSE', D
ATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'f
alse', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
3 row(s) in 0.0530 seconds
```

Which translates directing into a create table command on the destination cluster:

```
hbase(main):005:0> create 'my_table'
To do
```

### (Optional) Pre-split your regions on the destination cluster

To do.

## Running the migration

To run the migration, the premise is simple: copy up all the data, then copy up all the data that has changed since the job started. Then, cutover your application to use the configuration.

For our requirements, nothing particularly complex is required here and was all done with a bash script. Just store the time that the job was started, then run CopyTable for each table, then copy up the data that's changed since the job was started. Something like this:

```
TABLES="""
table1
table2
"""

STARTTIME_IN_MS=$(($(date +%s%N)/1000000))

for table in TABLES; do
   hbase org.apache.hadoop.hbase.mapreduce.CopyTable --new.name=$table $table
done

for table in TABLES; do
   hbase org.apache.hadoop.hbase.mapreduce.CopyTable --new.name=$table $table --starttime=STARTTIME_IN_MS
done
```

Then, once you've done some basic verification of the remote cluster (count rows etc), you're ready to cut your application over to the destination cluster.
