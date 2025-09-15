---
title: JSON-LD
date: 2025-09-14 00:00
modified: 2025-09-14 00:00
status: draft
tags:
- LinkedData
---

See [RDF Serialisation](rdf-serialisation.md).

JSON-LD is a format for expressing semantic, machine-readable data using a JSON syntax. It's used to represent [RDF](rdf.md) data on the web, making it easy for applications and search engines to understand and integrate linked data.

Example: Book Recrod

Step 1. JSON-LD Representation

```json
{
    "@context": {
        "schema": "https://schema.org/",
         "name": "schema:name",
         "author": "schema:author",
         "Book": "schema:Book",
    },
    "@type": "Book",
    "name": "The Great Gatsby",
    "author": {
        "@type": "schema:Person",
        "name": "F. Scott Fitzgerald"
    }
}
```

The `@context` tells us how to interpret as [Linked Data](linked-data.md).

Step 2. RDF Triples

When loaded into a semantic datbase, the JSON-LD is turned into **RDF Triples** (subject -> predicate -> object):
1. `_:book1` -> `rdf:type` -> `schema:Book`
2. `_:book1` -> `schema:name` -> `"The Great Gatsby"
3. `_:book1` -> `schema:author` -> `_:author1`
4. `_:author1` -> `rdf:type` -> `schema:Person`
5. `_:author1` -> `schema:name` -> `"F. Scott Fitzgerald"`

`_:book1` and `_:author` are blank nodes because we didn't give explicitly URIs.

Step 3. Stored in Semantic Database

In a triple store (like [GraphDB](GraphDB.md) and [Fuseki](Fuseki.md)).
* Each triple is stored in the graph.
* You can query it with [SPARQL](sparql.md).

For example:

```
SELECT ?book ?authorName WHERE {
    ?book a schema:Book ;
        schema:name "The Great Gatsby" ;
        schema:author ?author .
    ?author schema:name ?authorName.
}
```

Result:

| book    | authorName          |
| ------- | ------------------- |
| _:book1 | F. Scott Fitzgerald |

