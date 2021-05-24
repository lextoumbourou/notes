# Chapter 1: Getting Started

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

# Search In Depth

## finding exact values

* When working with exact values, you want to use filters. Filters are fast!
* The most basic type of filter is called a ``term`` filter.
* ``term`` filters can be used as follows:
```
{
    "query": {
        "filtered": {
           {
               "filter" : {
                   "term": {
                       "audience_gender": "male"
                   }
               }
           }
        }
    }
}
```
* That will do a filter and match only results where ``audience_gender`` is male.
* For items, like product ids that should always only match *exactly* you should setup a 'Customized field mapping' to ensure the field is not indexed.
```
{
    "mappings" : {
        "products" : {
            "properties" : {
                "productID" : {
                    "type" : "string",
                    "index" : "not_analyzed" 
                }
            }
        }
    }

}
```


* Results from ``GET /:index/:type/:id/_search`` are available in the ``hits`` array.
* Basic querying can be achieved with the ``q=<something>`` query param:
```
GET /lexandstuff/employees/_search?q=last_name:Toumbourou
{
   "took": 2,
   "timed_out": false,
   "_shards": {
      "total": 5,
      "successful": 5,
      "failed": 0
   },
   "hits": {
      "total": 1,
      "max_score": 0.30685282,
      "hits": [
         {
            "_index": "lexandstuff",
            "_type": "employees",
            "_id": "1",
            "_score": 0.30685282,
            "_source": {
               "first_name": "Lex",
               "last_name": "Toumbourou",
               "age": 27,
               "about": "I work hard.",
               "interests": [
                  "horror",
                  "hating"
               ]
            }
         }
      ]
   }
}
```

## Search with Query DSL

* The same query as above can be achieved with:

```
GET /lexandstuff/employees/_search
{
   "query": {
      "match": {
         "last_name": "Toumbourou",
      }
   }
}
```

* Can add a *filter* to a query to find employees older than 30

```
GET /lexandstuff/employees/_search
{
    "query": {
        "filtered": {
            "filter": {
                "range": {
                   "age": { "gt": 30}
                }
            },
            "query": {
                 "match": {
                     "last_name": "toumbourou"
                 }
            }
        }
    }
}
```

* Perform a full-text search, like so:

```
GET /lexandstuff/employees/_search
{
    "query": {
      "match": {
        "about": "number hater"
      }
    }
}
```

* Phrase searching can be performed with the "match_phrase" query:

```
GET /lexandstuff/employees/_search
{
    "query": {
      "match_phrase": {
        "about": "Number 1"
      }
    }
}
```

* To highlight matches snippets, use ``highlight`` param:

```
GET /lexandstuff/employees/_search
{
    "query": {
      "match_phrase": {
        "about": "Number 1"
      }
    },
    "highlight": {
      "fields": {
        "about": {}
      }
    }
}
...
{
    "_index": "lexandstuff",
    "_type": "employees",
    "_id": "3",
    "_score": 0.30685282,
    "_source": {
       "first_name": "Bobby",
       "last_name": "Goofs",
       "age": 31,
       "about": "Number 1 hater.",
       "interests": [
          "hating",
          "fucking"
       ]
    },
    "highlight": {
       "about": [
          "<em>Number</em> <em>1</em> hater."
       ]
    }
}
```

* To run Analytics against data, use the ``aggregrations`` function.

```
GET /lexandstuff/employees/_search
{
    "aggs": {
      "all_interests": {
        "terms": {"field": "interests" }
      }
    }
}
```

It will return an ``aggregations`` like so:

```
"aggregations": {
   "all_interests": {
      "buckets": [
         {
            "key": "hating",
            "doc_count": 3
         },
         {
            "key": "fucking",
            "doc_count": 2
         },
         {
            "key": "horror",
            "doc_count": 1
         }
      ]
   }
}
```

The agg can also include a "filter", to do stuff like, filter on people over 30 and find out what there interests are (who they be with...)

```
GET /lexandstuff/employees/_search
{
  "query": {
        "filtered": {
            "filter": {
                "range": {
                   "age": { "gt": 30 }
                }
            }
        }
    },
    "aggs": {
      "all_interests": {
        "terms": {
          "field": "interests"
        }
      }
    }
}
```
