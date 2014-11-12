# Chater 1: Getting Started

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
