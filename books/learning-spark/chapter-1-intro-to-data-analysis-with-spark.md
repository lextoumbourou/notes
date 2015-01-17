# Chapter 1: Introduction to Data Analysis with Spark

* Extends MapReduce and includes: interactive queries and stream processing
* Supports running computations in memory
* Responsible for scheduling, distributing and monitoring applications consisting of many computational tasks
  * Spark is fast and general-purpose

## Components of Spark

### Spark Core
  
* Task scheduling
* Memory management
* Fault recovery
* Interacting with storage systems
* Home to API that defines Resilient Distributed Datasets (RDD)
  * Collection of items distributed across many nodes that can be manipulated in parallel

### Spark SQL

* Allows for querying data via SQL (and Hive Query Language (HQL))
* Shark is an older SQL-on-Spark project that has been replaced by Spark SQL

### Spark Streaming

* For processing live streams of data (like log files)

### MLlib

* Provides multiple types of Machine Learning algorithms
  * Classification (unsupervised)
  * Regression (supervised)
  * Clustering
  * Collaborative filtering
* Provides lower-level ML primitives including a generic gradient descent optimization algorithm

### GraphX

* Library for manipulating graphs (like a social network friend graph)

### Cluster Managers

* Spark can run over many cluster managers like Hadoop YARN, Apache Mesos or the cluster manager included in Spark called Standalone Scheduler.

### History

* Started in 2009 as research project
* Transferred to Apache Foundation in 2013
