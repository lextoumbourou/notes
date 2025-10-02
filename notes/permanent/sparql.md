---
title: SPARQL
date: 2025-09-16 00:00
modified: 2025-09-16 00:00
summary: "a query language for RDF databases"
aliases:
- RDF Query Language
---

**SPARQL** (or **SPARQL Protocol and RDF Query Language**) is a Query Language for [RDF](rdf.md) databases and [Triplestores](triplestores.md).

It uses pattern-based queries. ph databases have no fixed starting points of order.
Use variables beginning with question marks (e.g., `?friend`, `?name`). It also defined how to communicate with SPARQL endpoints over HTTP - it's not just a query language.

## Example Queries

All of these queries can be run live against https://dbpedia.org/sparql

---

Let's start with a query to fetch basic info about the city of London. We can define variables using a `?` prefix. Then in our query we can describe a pattern which matches [RDF Triples](rdf-triples.md) where the Subject is [London](https://dbpedia.org/page/London) and any predicate or object matching.: `<http://dbpedia.org/resource/London> ?property ?value .`

Also, for readability, we'll convert `http://dbpedia.org/resource/` into a prefix:

```sparql
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?property ?value
WHERE {
    dbr:London ?property ?value .
} LIMIT 20
```

**Challenge**: Can you list basic information about Sydney, Australia?

Answer:

```sparql
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?property ?value
WHERE {
    dbr:Australia ?property ?value .
} LIMIT 20
```

---

We define much more complex patterns for the data we want to fetch by providing multiple triples statements. The RDFS schema also contains a number of useful predicates, like `rdf:type` (which is used to say subject isA object) and `rdfs:label` is a human readable name for a resource.

Here's a query that gets all the cities in the UK (but only the English labels):

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?city ?name
WHERE {
    ?city dbo:country dbr:United_Kingdom .
    ?city rdf:type dbo:City .
    ?city rdfs:label ?name .
    FILTER (lang(?name) = 'en')
}
```

You can rewrite the above query is saving having to use the `?city` prefix on each line as follows. Also, an alias for `rdf:type` is `a`:

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?city ?name
WHERE {
    ?city dbo:country dbr:United_Kingdom ;
          a dbo:City ;
          rdfs:label ?name .
    FILTER (lang(?name) = 'en')
}
```


Challenge: using `dbo:TelevisionShow` as the type, and using `dbo:genre dbo:Science_fiction` return all science fiction TV Shows from the produced or created in Australia.

Answer:

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?show ?name
WHERE {
    ?show dbo:genre dbr:Science_fiction ;
          dbo:country dbr:Australia ;
          a dbo:TelevisionShow ;
          rdfs:label ?name .
    FILTER (lang(?name) = 'en')
}
```

---

### Filtering

We can use the `FILTER` call to filter by certain values. For example, to find all cities in the world that have more than 10_000_000 people, and you can `ORDER BY DESC(?population)`

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?city ?name ?population
WHERE {
    ?city a dbo:City ;
          rdfs:label ?name ;
          dbo:populationTotal ?population .
    FILTER (lang(?name) = 'en')
    FILTER (?population > 10000000)
}
ORDER BY DESC(?population)
LIMIT 10
```

Challenge: the album genre of rock music is `dbp:genre dbr:Rock_music`, the property of release year is available in `dbp:relYear` and the `rdf:type` is `dbo:Album`. Can you list all the rock albums released after 2010?

Answer:

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX dbp: <http://dbpedia.org/property/>

SELECT ?album ?name ?relyear
WHERE {
    ?album a dbo:Album ;
            dbp:genre dbr:Rock_music ;
            rdfs:label ?name ;
            dbp:relyear ?relyear .
    FILTER (lang(?name) = 'en')
    FILTER (?relyear > 2015)
}
ORDER BY DESC(?relyear)
LIMIT 10
```