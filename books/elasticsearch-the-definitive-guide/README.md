# Getting Started

## Setup

* Download and install ElasticSearch like:

```
curl --location --remote-name https://download.elasticsearch.org/elasticsearch/elasticsearch/elasticsearch-1.3.4.zip
unzip elasticsearch-1.3.4.zip
cd elasticsearch-1.3.4
```

* Install Marvel, a management and monitoring tool for Elasticsearch.

```
./bin/plugin -i elasticsearch/marvel/latest
echo 'marvel.agent.enabled: false' >> ./config/elasticsearch.yml
```

* Run Elasticsearch like (`-d` = daemon mode)

```
./bin/elasticsearch -d
```

##  Indexing

* Relational DB:
  * Relational DB -> Databases -> Tables -> Rows -> Columns
* Elasticsearch:
  * Elasticsearch -> Indices -> Types -> Documents -> Fields
* Count all documents in cluster

```
GET /_count
{
   "query": {
      "match_all": {}
    }
}
```
* No need to create indexes or specify type of data fields contains:
  * Request:
  ```
  PUT /bickle/employee/1
  {
    "first_name": "Travis",
    "last_name": "Bickle",
    "age": 25,
    "about": "You talking to me?",
    "interests": ["taxis", "betty"]
  }
  ```
  * Response:
  ```
  {
     "_index": "bickle",
     "_type": "employee",
     "_id": "1",
     "_version": 1,
     "created": true
  }
  ```
* Can retrieve documents like so:
```
> curl -XGET 'http://localhost:9200/bickle/employee/1'
{"_index":"bickle","_type":"employee","_id":"1","_version":1,"found":true,"_source":{
  "first_name": "Travis",
  "last_name": "Bickle",
  "age": 25,
  "about": "You talking to me?",
  "interests": ["taxis", "betty"]
}
```

## Searching

* Can search like so:
```
> curl -XGET 'http://localhost:9200/bickle/employee/_search'
{"took":7,"timed_out":false,"_shards":{"total":5,"successful":5,"failed":0},"hits":{"total":2,"max_score":1.0,"hits":[{"_index":"bickle","_type":"employee","_id":"1","_score":1.0,"_source":{
  "first_name": "Travis",
  "last_name": "Bickle",
...
```
* Specify a filter param like:
```
> curl -XGET 'http://localhost:9200/bickle/employee/_search?q=first_name:Travis'
```
* Or use query dsl:
```
> curl -XGET 'http://localhost:9200/bickle/employee/_search' -d '
{
  "query": {
      "match": {
         "last_name": "Bickle"
       }
   }
}
'
```

## Questions at this point

* How often should we index data?
* Can we automatically index our db?
