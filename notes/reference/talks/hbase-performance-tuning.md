## Hbase Performance Tuning

* HDFS
  * Config: ``hdfs-site.xml``.
  * Stores Hbase WAL and HFiles.
  * No sync-to-disk by default.
  * Datanode writes ``tmp`` file, moves it into place.
  * Old data lost on power outage.
    * Always set to avoid this: ``dfs.datanode.synconclose = true``
      * Will make stuff sloooow.
    * To get around the slowness:
      * "HDFS Sync Behind Writes"
        * Necessary with sync-on-close for performance.
      * ``dfs.datanode.sync.behind.writes``
      * "Datanode will instruct the OS to enqueue all written data to disk immediately after it is written."
  * Stale Datanodes
    * Don't use a DN for read and write when it looks like it is stale:
      * ``dfs.namenode.avoid.read.stale.datanode = true``
      * ``dfs.namenode.avoid.write.stale.datanode = true``
      * ``dfs.namenode.state.datanode.interval = 30000`` (30 seconds)
  * HDFS short circuit reads
    * Read local blocks directly without DN, when RegionServers and DN are co-located.
    * ``dfs.client.read.shortcircuit = true``
    * ``dfs.client.read.shortcircuit.buffer.size = 131072`` - (default)
      * Hbase keeps a reader open for every column, keep it small.
    * ``hbase.regionserver.checksum.verify = true``
    * ``dfs.domain.socket.path``
  * Misc HDFS tips: ``dfs.datanode.failed.volumes.tolerated = <N>`` (tolerate losing that many disks).
  * Distribute data across disks at a DN
    * ``dfs.datanode.fsdataset.volume.choosing.policy = AvailableSpaceVolumnChoosingPolicy``
  * Misc HDFS settings
    * ``dfs.block.size = 268435456`` (256M ideal)
    * ``ipc.server.tcpnodelay = true`` (turn off Nagos algorithm)
    * ``ipc.client.tcpnodelay = true``
    * ``dfs.datanode.max.xcievers = 8192``
    * ``dfs.namenode.handler.count = 64``
    * ``dfs.datanode.handler.count = 8`` (match number of spindles)

* Hbase
  * Compactions (background)
    * Writes are buffered into memstore
    * Memstore contents flushed to disk as HFiles.
    * Limit # Hfiels by rewriting small HFiles into larger ones
    * Remove deleted and expired Cells
    * Same data written multiple times -> write amplification
    * Read vs Write
      * Read requires merging HFiles -> fewer HFiles better.
      * Write throughput better with fewer compactions -> leads to more files
      * ``hbase.hstore.blockingStoreFiles = 10``
        * Small for reads, large for writes (large for bursty writes)
      * ``hbase.hstore.compactionThreshold = 3``
        * Small for read, large for writes
      * ``hbase.hregion.memstore.flush.size`` - larger good for read and writes, but need to keep heap in mind.
      * ``hbase.hregion.majorcompaction = 604800000`` (once a week by default)
      * ``hbase.hregion.majorcompaction.jitter = 0.5`` (1/2 week by default)
    * Memstore / Cache sizing
      * ``hbase.hregion.memstore.flush.size = 128``
      * ``hbase.hregion.memstore.block.multiplier``
        * Allow memstore to grow by this multiplier. Good for heavy bursty writes.
      * ``hbase.regionserver.global.memstore.size`` (decrease for read heavy load, increase for write heavy).
      * ``hfile.block.caches.size`` (percent heap used for block cache)
      * ``hbase.regionserver.global.memstore.size``
  * Data Locality
      * ``hbase.hstore.min.locality.to.skip.major.compact``
      * ``hbase.master.wait.on.regionservers.timeout``
      * Don't use the HDFS balancer??
  * HFile Block size
    * Don't confuse with HDFS block size
  * Garbage Collection
    * Works based on "Weak Generational Hypothesis"
      * Most allocated objects die young
      * HotSpot manages 4 generations
        1. Eden for all new objects
        2. Survivor 1 and 2 where surviving objects are promoted when Eden is collected.
        3. ``Tenured`` space: objects surviving a few rounds (16 by default) of Eden/Survivor collection are promoted into the tenured space.
        4. Perm gen (classes, intered strings) and more or less permanent object (gone in JDK8).
      * Memstore is relatively long-lived.
      * Blockcache is long-lived (allocation in 64k blocks)
      * Deal with "operational" garbage
      * Super small young-gen (``Xmn512m``)
      * Collect Eden in Parallel (``-XX:+UseParNewGC``)
      * Use the non-moving CMS collector (``-XX:+UseConcMarkSweepGC``)
      * Start collecting when 70% of tenured gen is full, avoid collection under pressure (``-XX:CMSInitiatingOccupanyFraction=70``)
      * Do not try to adjust CMS setting (``-XX:+UseCMSInitiatingOccupancyOnly`)

  * RegionServer Machine Sizing
    * Disk/Java Heap Ratio
      * RegionSize / MemstoreSize * ReplicationFactor * HeapFractionForMemstore * 2
      * ``hbase.regionserver.maxlogs`` - should be larger than 40% of your heap. Don't flush of this setting.
    * Hbase is CPU intensive on read, IO heavy on write.
  * Linux
    * Turn THP (Transparent Huge Pages) OFF
    * Set Swappiness to 0
    * Set ``vm.min_free_kbytes`` to AT LEAST 1GB (8GB on larger systems)
