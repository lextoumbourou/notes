# Redis

## Programming with Redis

### Using Redis as LRU (less recently used) cache

* Eviction policies
  * ``noeviction`` - return errors when ``maxmemory`` is reached.
  * ``allkeys-lru`` - evict keys when we hit ``maxmemory`` trying to remove the less recently used keys first.
  * ``volatile-lru`` - same as above, but only remove keys that have an *expire set* configured.
  * ``allkeys-random`` - evict keys randomly.
  * ``volatile-random`` - evict keys randomly which have an *expire set* configured.
  * ``volatile-ttl`` - evict keys with a shorter ttl that have an *expire set* configured.
* General eviction policy guidelines
  * Use ``allkeys-lru`` when you expect a subset of elements will be accessed more than the rest (a [power law](http://en.wikipedia.org/wiki/Power_law) distribution). A good pick if unsure.
  * Use ``allkeys-random`` if all keys are scanned continuously, or all keys are accessed with equal probablility (normal distribution).
  * Use ``volatile-ttl`` if application provides hints about what to expire.
* How eviction policy works
  * Client runs a command and more data is added to Redis.
  * Redis checks mem usage, and if ``maxmemory`` is set, it evicts keys according to policy.
* Tuning samples to check for eviction
  * Redis' LRU implementation is not "exact".
  * Tune ``maxmemory-samples`` (at expense of CPU) to tune precision of algorithm.

## Administration

### Redis configuration

* To configure Redis as just a cache, use the ``maxmemory-policy allkeys-lru`` policy:

```
maxmemory 2mb
maxmemory-policy allkeys-lru
```

### Redis persistence

* Options:
  * RDB
    * performs point-in-time snapshot of data at configurable intervals.
    * pros
      * compact (good for backups).
      * good for disaster recovery.
      * maximizes Redis performance.
      * allows for faster restarts.
    * cons
      * not good for minimizing chance to data loss
      * RDB needs to ``fork()`` to persist on disk with a child process - can be time consuming if dataset is big (may block clients)
  * AOF
    * logs every write operation received by server and plays it at server start up.
    * pros
      * durable
      * append only log, so no seeks nor corruption problems if outage.
      * Redis can rewrite AOF log in background when it gets too big.
    * cons
      * bigger files
      * can be slower than RDB
  * None - disable persistance and delete everything when Redis is restarted.
* Snapshotting
  * Redis saves to ``dump.rdb`` even N seconds if there are at least M changes to dataset:
    ```
    save N M
    ```
  * Append-only file can be configured if more durablility is required with ``appendonly yes``
