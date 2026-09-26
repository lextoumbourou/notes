---
title: Wikidata
date: 2025-08-23 00:00
modified: 2026-09-26 08:55
status: draft
---

Example of Wikidata [SPARQL](sparql.md) queries:

Get all humans (limit 10)

```sparql
SELECT * WHERE {
    ?thing wdt:P31 wd:Q5
} LIMIT 10
```

More here: <https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries#Optionality_and_alternatives>

Get all humans where the Twitter username is ….

```sparql
SELECT * WHERE {
    ?thing wdt:P31 wd:Q5
    ## I do something but I don't know what?
} LIMIT 10
```
