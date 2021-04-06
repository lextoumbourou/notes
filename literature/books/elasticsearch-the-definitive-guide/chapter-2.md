# Chapter 2: Life Inside A Cluster

* Node running single instance of ES is an *empty cluster*.
* When in cluster mode, one node is elected to be *master* node, in charge of managing cluster-wider system changes.
  * creating or deleting index
  * adding or removing nodes from cluster
  * master node does not need to be involved in document changes or search, so isn't a bottle neck.
* any node can become master

## Cluster heath

* Check cluster health as follows:

```
GET _cluster health
{
    "cluster_name": "elasticsearch",
    "status": "green",
    "timed_out": false,
    "number_of_nodes":
    "number_of_data_nodes":  1,
    "active_primary_shards": 0,
    "active_shards":         0,
    "relocating_shards":     0,
    "initializing_shards":   0,
    "unassigned_shards":     0
}
```

* ``status`` field provides overall indiciation of cluster health:
  * ``green`` - all primary and replica shards are active
  * ``yellow`` - all primary shards are active, but not all replicas are active
  * ``red`` - not all primary shards are active

## Add an index

* Index is a *logical namespace* that maps to one or more physical shards.
* Shards
 *  Low-level *worker unit* which holds some of the data in an index.
  * Single instance of Lucene and a complete search enginer
  * As ES grows or shrinks, it will auto migrate shards between nodes
  * Shards can be primary or replica
    * Replica is a copy of data to protect against hardware failure
    * Can be used to serve read request like searching or retrieving a doc

## Add failver

* Just start a second node with same ``cluster.name`` as first node in ``config/elasticsearch.yml``.
  * Requests ``multicast`` to communicate

## Scale horizontally

To add more replica shards:

```
PUT /blogs/_settings
{
    "number_of_replicas" : 2
}
```
