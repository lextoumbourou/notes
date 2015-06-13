# Lesson 2: HDFS and Mapreduce

* HDFS (Hadoop Distributed File System)
    * Makes distributed filesystem look like a regular filesystem
    * Breaks files down into blocks
    * Distributes blocks to different nodes in the cluster based on algorithm (which algorithm ?)
    *  Name-node (single node) is used to distribute data
* Data redundancy
    * Stores copy of one block on data on 3-nodes 
    * Block size is 64MB by default
    * Solves problems:	 
        * Network failures can cause missing data
        * Failures individual nodes can "break" data
* NameNode standby
    * Namenode was initially a massive single point of failure: if it breaks, data can be lost
    * Problem solved with Active/Standby
        * When Active fails, standby picks up the job
* HDFS Commands
    * Mirrors Linux equivalents
    * Examples:	 
        * list contents of HDFS with ls				
        ```	
        > hadoop fs -ls
        ```		
        * puts file into HDFS with put				
        ```
        > hadoop fs -put purchases.txt
        ```		
        * display last few lines with tail		
        ```
        > hadoop fs -tail purchases.txt
        ```	
        * remove file with rm	
        ```
        > hadoop fs -rm
        ```	
        * display entire contents of file with cat		
        ```
        > hadoop fs -cat purchases.txt
        ```
* Mapreduce is designed to process in parallel
* Mappers
    * Deal with a small amount of data in parallel
    * Puts into intermediate records (hashtable-like key/value structure)
    * Shuffle-and-sort into reducer
* Reducers
    * Collect the various hashtables and processes them (perhaps adds them)
* Multiple reducers
    * Generally, you don't know which keys are going to which reducer
* Daemons of MapReduce
    * Data nodes
        * Task tracker
            * Actually runs the map/reduce code 
            * Input split is concept of trying to ensure task tracker runs on nodes with data (kinda).
    * Name nodes
    * Job tracker
        * To learn: does this run on the name node?
* Running a job
    * Hadoop streaming let's you use any language to write mapper/reducer
    * Output directory must not already exist to run your jobs
    * Example:

        ```
        > ls
        mapper.py reducer.py
        > hadoop jar /path/to/hadoop-streaming-mr1-cdh4.1.1.jar \
        -mapper mapper.py -reducer.py \
        -file mapper.py -file reducer.py \
        -input myinput -output joboutput
        ```
