# Databases 101

* Relational Database (RDS)
  * SQL Server
  * Oracle
  * MySQL Server
  * PostgreSQL
  * Aurora
  * MariaDB

* Non-Relational Database
  * Collection (table)
  * Document (row)
  * Key-value pairs (fields)

* Data Warehousing
  * Used for "business intelligence"
  * Tools like Cognos, Jaspersoft, SQL Server Reporting Services, Oracle Hyperion, SAP NetWeaver
  * Use to pull in large and complex data sets
  * Used by management to do queries on data (current performance vs targets)
  * Often a copy of your database to run complex queries away from customer traffic.
  * OLTP vs OLAP
    * Online Transaction Processing (OLTP)
      * Example: pull up an order based on some number and get data associated with it.
    * OLAP Online Analytics (OLAP)
      * Example: find net profit for division of a company in a region.

* Elasticache
  * Memcache
  * Redis

* Database Migration Service (DMS)
  * Let's you migrate data from one platform to AWS.
  * Also lets you migrate from one DB type to another.

## RDS

* Backups
  * 2 types: automated and db snapshots.
  * Automated
    * Recover your data to a point in time in a "retention period"
    * Retention period can be between one and 35 days.
    * Can store transaction logs throughout the day, to do a point in time recovery down to a second within the retention period.
    * Enabled by default.
    * Backup data is stored in S3 and you get space equal to size of DB free.
      * Eg if you have 10G of data, you get 10G of free space.
    * During backup window, I/O may be suspended while the data is backed up.
    * May experience elevated latency.
  * Snapshots
    * Must be done manually.
    * Snapshots are stored even after RDS instance is deleted.

* Encryption
  * Supported for all except Aurora.
  * Once encrypted, all data, replicas, backups and snapshots are encrypted too.
  * Cannot encrypt an existing database.

* Multi-AZ RDS
  * Have exact copy in another AZ.
  * Can fail over automatically.

* Read Replica
  * Use to boost performance of read-heavy workloads.
  * Supported on MySQL Server, PostgreSQL and MariaDB (not SQL or Oracle).
  * Must have auto backups on.
  * Can have up to 5 read replicas of a DB.
  * Can have read reps or read reps.
  * Each rep will have a DNS end point.
  * Cannot have multi-AZ read reps but can create read reps of multi-AZ source DBs.
  * Read reps can be promoted to own DB.

* DynamoDB vs RDS
  * DyanmoDB offers "push button" scaling, can scale without downtime.
  * RDS not so easy.

## DynamoDB

* Most import for "certified developer exam".
