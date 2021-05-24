# Chapter 6: Mapping and analysis

* When data is index, ES takes the string value of all its fields and chucks them into a big field called ``_all``. That's what's queried when you use the ``?q`` param without specifying a field.

* Fields can be indexed as "exact values" or as "full text".

* Full text
  * Understands intent
    *  (search for USA should match United States of America).
    * "jump" should also match "jumped", "jumps" etc
  * ES analyzes the text then uses the results to build "inverted index"

* Inverted index is created by spliting the "content" field of each document into separate words (or *terms* or *tokens*)

```
Term   id_1    id_2
-------------------
hello   x
world           x
yo      x
```

* Analysis is the process of:
  * tokenizing a block of text into individual "terms" for use in inverted index (example above)
  * "normalise" terms to improve searchabilty
    * The query must normalise both the query and the index in the same form.

* This job is performed by "analyzers", which is a wrapper that combines 3 functions into a single package:
  1. Character filters
    * String is passed through any *character filters* in turn. Job is to tidy up the string before tokenization. Could strip out HTML and convert &s to "and".
  2. Tokenizer
    * Split string into individual terms by splitting up text when it finds whitespace or punctuation
  3. Token filters
    * Do stuff like lowercasing terms, removing stop words like "a", "and", "the" or adding synonyms like "jump" and "leap".

* Built-in analyzers

  * Standard analyzer

      * Default choice.
      * Splits text on "word boundaries" defined by Unicode Consortium.
      * Removes most punctuation.
      * Lowercases all terms.
      * Result:

       ```Set the shape to semi-transparent by calling set_trans(5)```

       ```set, the, shape, to, semi, transparent, by, calling, set_trans, 5```

  * Whitespace analyser

    * Splits text on whitespace.
    * Doesn't lowercase.
    * Result:

      ```Set the shape to semi-transparent by calling set_trans(5)```

      ```Set, the, shape, to, semi-transparent, by, calling, set_trans(5)```

  * Language analyzer

    * Can take peculiarities of languages into account.
    * ```english``` analyzer comes with a set of English stopwords.
    * Words like ```and``` or ```the``` are removed and can ```stem``` English words.

* Analyzers occur during both index and on searches and ES automatically performs the same analysis on queries as the fields searched.
  * When full-text is queried: query is analysed the same as field.
  * When exact match is queried: no analysis is performed.

* Use ```analyze``` API to see how text is analysed.

```
GET /_analyze?analyzer=standard
{
  "What's up"
}
```

```
{
   "tokens": [
      {
         "token": "what's",
         "start_offset": 5,
         "end_offset": 11,
         "type": "<ALPHANUM>",
         "position": 1
      },
      {
         "token": "up",
         "start_offset": 12,
         "end_offset": 14,
         "type": "<ALPHANUM>",
         "position": 2
      }
   ]
}
```

  * ```token``` - The actual token stored in the index.
  * ```start_offset``` & ```end_offset``` - The character positions that the original word occupied.
  * ```position``` - The order in which is appeared in the doc.
  * ```type``` - vary per analyzer. Can be ignored.

* Can specify analyzer for a field using the ```mapping``` API.

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
