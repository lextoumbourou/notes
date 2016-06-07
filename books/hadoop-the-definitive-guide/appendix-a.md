# Appendix A. Install Apache Hadoop

```
> wget http://apache.mirror.digitalpacific.com.au/hadoop/common/hadoop-2.7.2/hadoop-2.7.2.tar.gz
> export HADOOP_HOME=~/bin/hadoop-2.7.2
> export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
> hadoop version
Hadoop 2.7.1
Subversion https://git-wip-us.apache.org/repos/asf/hadoop.git -r 15ecc87ccf4a0228f35af08fc56de536e6ce657a
Compiled by jenkins on 2015-06-29T06:04Z
Compiled with protoc 2.5.0
From source with checksum fc0a1a23fc1868e4d5ee7fa2b28a58a
This command was run using /usr/local/Cellar/hadoop/2.7.1/libexec/share/hadoop/common/hadoop-common-2.7.1.jar
```

* Each Hadoop property configured with XML file.
  * Common properties = ``core-site.xml``
  * Properties shared between HDFS, MapReduce and YARN go in ``hdfs-site.xml``, ``mapred-site.xml`` and ``yarn-site.xml``
    * All located in ``etc/hadoop`` subdirectory.

* Can be run in 3 modes:
  * Standalone (or local) mode.
    * No daemons, everything in a single JVM.
    * Suitable for testing MapReduce problems.
  * Pseudodistributed mode:
    * Hadoop daemons run on local machine, simulating small cluster.
  * Fully distributed mode:
    * Normal way to operate in prod.

* Need to enable password less SSH access to current user (System Preferences -> Sharing -> Remote Login in OSX)

* Format HDFS filesystem:

```
> hdfs namenode -format
```

* Start and stop daemons:

```
> start-dfs.sh  # Starts namenode, secondary namenode and datanode.
> start-yarn.sh  # starts resource manager and node manager.
> mr-jobhistory-daemon.sh start historyserver
```

* Create files:

```
> export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
> hdfs dfs -mkdir -p /user/$USER
```
