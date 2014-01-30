# Lesson 1: Big Data

## Notes

* Data Sources
	* 90% of world's data was created in last 2 years alone
	* Cell tower data (where signal strength is highest etc)
	* Website logs
* What is Big Data?
	* Subjective term
	* Generally described as "data too big to fit on one machine"
* Challenges
	* Data is created rapidly
	* Data in a range of formats	 	
* The 3 Vs
	* Volume - size of data
	* Variety - different formats
	* Velocity - speed it's being generated
* Using Data
	*  eCommerce - 5 minutes looking at item, send you an item to tell when item is on sale.
	*  NetFlix - based on your viewing habits, they can recommend films to you. (shout outs to [Hackers](http://www.imdb.com/title/tt0113243/) reference)
* Doug Cutting (one of the founders of Hadoop) Intro
	* Working on open-source search engine called "Nutch"
	* Read Google's paper on MapReduce
	* His colleague went to try to implement it in open-source project called Hadoop
	* Yahoo invested in it and he worked on it at Yahoo
	* Languages to query it like Pig
* Core Hadoop
	* Store in HDFS
		* Distributed filesystem 
	* Process with MapReduce  
		* Process HDFS as if it was on a single server
* Hadoop Ecosystem
	* Hive
		* let you write SQL that gets turned into Map/Reduce code
	* Pig
		* Another query language converted to Map/Reduce
	* Impala
		* Query your data with SQL directly accessing HDFS (bypassing compile to Map/Reduce)
	* Sqoop/Flume
		* Puts data into cluster in relational db format
	* HBase
		* Real-time DB built on HDFS 

## Takeaway

* Businesses should aim to store as much data as they can. Very little of it is worthless.
* Hadoop/HDFS has a lively ecosystem

## Opinion

Pretty simple stuff. Nothing I haven't heard before. Nevers hurts to hear the basics again though. :)