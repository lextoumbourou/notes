# Chapter 16: Partial matching

## Ngrams for partial matching

* n-grams can be thought of as "moving windows on a word". 
* To n-gram the world "hello", the results would depend on length chosen.
  * Length 1 (unigram) ['h', 'e', 'l', 'l', 'o']
  * Length 2 (bigram) ['he', 'el', 'll', 'lo']
  * Length 3 (trigram) ['hel', 'ell', 'llo']
  * Length 4 (four-gram) ['hell', 'ello']
* Plain n-grams can be used for "somewhere in a word" searches.
* For "search-as-you-type" you should use a specialise form of n-gram called "Edge n-gram", which are anchored in beginning of word. Eg:
  * h
  * he
  * hel
  * hell
  * hello
* To prepare the index, first create an autocomplete filter:
```
{
  "filter": {
    "autocomplete_filter": {
      "type": "edge_ngram",
      "min_gram": 1,
      "max_gram": 20
    }
  }
}
```
In English: for any term this token filter receives, produce an n-gram anchored to start of word with min length 1 and max length 20.
* Then, you need to use filter in custom analyzer, called ``autocomplete`` analyzer.
```
{
  "analyzer": {
    "autocomplete": {
      "type": "custom",
      "tokenizer": "standard",
      "filter": [
        "lowercase",
        "autocomplete_filter"
      ]
    }
  }
}
```
* The full command for creating this is as follows:
```
PUT /my_index
{
  "settings": {
    "number_of_shards": 1,
      "analysis": {
        "filter": {
            "autocomplete_filter": {
              "type":     "edge_ngram",
              "min_gram": 1,
              "max_gram": 20
            }
          },
          "analyzer": {
            "autocomplete": {
              "type":      "custom",
              "tokenizer": "standard",
              "filter": [
                "lowercase",
                 "autocomplete_filter"
              ]
          }
        }
      }
  }
}
```
* To use analyzer, it needs to be applied to a field. That can be done with the "upate-mapping" API:
```
PUT /some_index/_mapping/some_type
{
    "some_type": {
        "properties": {
          "name": {
            "type": "string",
            "analyzer": "autocomplete"
          }
       }
    }
}
```
* Now the field can be queried using a simple match query:
```
GET /some_index/some_type_search
{
    "query": {
        "match": {
          "name": "fo"
        }
     }
}
```
* Because we have only specified "analyzer" in the field properties, the "autocomplete" analyzer will be used for both tokenizing and searching. Which will mean a search for "USA", will also search for "U", "S", "A" - not good. What we should do is use the "autocomplete" analyzer for indexing and the "standard" analyzer for searching.
```
PUT /some_index/_mapping/some_type
{
    "some_type": {
      "properties": {
        "name": {
          "type": "string",
          "index_analyzer": "autocomplete"
          "search_analyzer": "standard"
        }
     }
  }
}
```
