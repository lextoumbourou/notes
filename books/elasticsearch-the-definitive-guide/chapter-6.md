# Chapter 6: Mapping and analysis

* To use ElasticSearch to its full potential, you need to understand 3 concepts
  * Mapping - how daa in each field is interpreted
  * Analysis - how full text is processed to make it searchable
  * Query DSL - flexible, powerful query language

# Chapter 6: Mapping and analysis

* Can learn how ElasticSearch has interpreted a document by requesting the *mapping* (or schema definition), like this:

```
GET /lexandstuff/_mapping/employees
{
   "lexandstuff": {
      "mappings": {
         "employees": {
            "properties": {
               "about": {
                  "type": "string"
               },
               "age": {
                  "type": "long"
               },
               "first_name": {
                  "type": "string"
               },
               "interests": {
                  "type": "string"
               },
               "last_name": {
                  "type": "string"
               }
            }
         }
      }
   }
}
```

* Data in ES is divided into two types: exact values and full text
* Inverted index is created by spliting the "content" field of each document into separate words (or *terms* or *tokens*)
```
Term   id_1    id_2
-------------------
hello   x
world           x
```
* Analysis is the process of:
  * tokenizing a block of text into individual "terms" for use in inverted index (example above)
  * "normalise" terms to improve searchabilty
* This job is performed by "analyzers", which is a wrapper that combines 3 functions into a single package:
  1. Character filters
    * String is passed through any *character filters* in turn. Job is to tidy up the string before tokenization. Could strip out HTML and convert &s to "and".
  2. Tokenizer
    * Split string into individual terms by splitting up text when it finds whitespace or punctuation
  3. Token filters
    * Do stuff like lowercasing terms, removing stop words like "a", "and", "the" or adding synonyms like "jump" and "leap".
