# EC2 101

* Amazon Elastic Compute Cloud (Amazon EC2)
* EC2 Options:
  * On Demand - fixed rate by the hour, no commitment.
    * No up-front payment or long-term commitment.
    * Short term, spiky, unpredicatable workloads.
  * Reserved - significant discount on hourly charge for an instance. 1 Year or 3 Year Terms.
    * Applications with steady state or predicable usage.
    * Upfront payments to reduce total costs.
  * Spot - enable you to bid whatever price you want for instance capacity.
    * When spot price meets your reserve, spin up the box.
    * When price is above your reserve, you get 1 hours notice before the box is terminated.
    * If spot is terminated by EC2, you aren't charged for the hour. If you terminate it, they'll charge you.
* EC2 instance types:
  * T2
    * Lowest cost, general purpose.
    * Use case: web servers / small DBs.
  * M4
    * General purpose.
    * App servers.
  * M3
    * General purpose.
    * App servers.
  * C4
    * Compute Optimize.
    * CPU intensive Apps/DBs.
  * C3
    * Compute Optimize.
    * CPU intensive Apps/DBs.
  * R3
    * R == RAM.
    * Memory Optimized.
    * Memory Intensive Apps/DBs.
  * G2
    * Graphics / General Purpose GPU.
    * Video Encoding / Machine Learning / 3D App Streaming
  * I2
    * I == IOPS.
    * High speed storage.
    * NoSQL DBs, Data Warehousing etc.
  * D2
    * D == Density.
    * Dense storage.
    * Fileservere, Data Warehousing, Hadoop etc.
* What is EBS?
  * "A disk in the Cloud"
  * Create storage and attach to EC2 instances.
  * Can't be shared between machines.
* EBS Volume Types
  * IOPS == input / output per second.
  * General Purpose SSD (GP2)
    * Designed for 99.999% availability.
    * Ration of 3 IOPS per GB with up to 10,000 IOPS
      * Can burst to 3000 IOPS for short periods for volumns under 1G
  * Provisioned IOPS SSD (IO1)
    * Designed for I/O intensive apps like large NoSQL databases.
    * Use when you need > 10000 IOPS.
  * Magnetic (Standard)
    * Lowest code per GB of all EBS.
    * Use when data access infrequently.
* Exam tips:
  * Know diff between On Demand, Spot, Reserved.
  * Know who pays when spot instances are terminated.
  * EBS types: General Purpose, Provisioned IOPS, Magnetic.
  * Can't mount 1 EBS across 2 instances, need to use EFS.

# Lab Summary

* Termination protection = off by default.
* Default action of root EBS is termination by default.
* Can't encrypt root volume of EBS disk.
