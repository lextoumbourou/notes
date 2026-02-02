---
title: RDFS
date: 2025-08-23 00:00
modified: 2025-08-23 00:00
status: draft
aliases:
- RDF Schema
---

**RDFS** or **RDF Schema** is an extension to the [RDF](rdf.md) document model that provides the basic structures for describing data structures and inheritance.

## Classes and Inheritance

For example, we can declare classes using `rdfs:Class` type:

```turtle
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

ex:DomesticatedMammal rdfs:type rdfs:Class .
```

We can support inheritance relationships through `rdfs:subClassOf` type, to create hierarchical taxonomies:

```turtle
ex:Dog rdfs:subClassOf ex:DomesticatedMammal .
ex:Poodle rdfs:subClassOf ex:Dog .
```

Therefore, every Dog is also a Domesticated Mammal, and every Poodle is both a Dog and a Domesticated Mammal. You can even subclass from multiple parent classes to create rich, interconnected vocabularies.

In [Lexical Semantics](lexical-semantics.md), `rdfs:subClassOf` captures the [Hyponym](hyponym.md) -> [Hypernym](hypernym.md) relationship between terms.

## Established Vocabularies

RDFS uses prefixes to make URIs more readable and leverages existing ontologies. For example, FOAF (Friend of a Friend) is a well-established ontology for describing social networks and contact details.

```turtle
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:lex rdfs:type foaf:Person .
ex:lex foaf:name "Lex Toumbourou"
ex:lex foaf:mbox <mailto:lextoumbourou@gmail.com>
```

## Properties and Relationships

Properties in RDFS define the relationship between resources and can serve as predicates in RDF triples. RDFS provides a powerful way to constrain and organise these properties.

Properties can specific their domain (what type of subject they can have) and range (what type of object they can have):

```turtle
ex:teaches rdfs:type rdf:Property .
ex:teaches rdfs:domain ex:Professor .
ex:teaches rdfs:range ex:Course .

ex:enrolledIn rdfs:type rdf:Property .
ex:enrolledIn rdfs:domain ex:Student .
ex:enrolledIn rdfs:range ex:Course .
```

Here only Professors can use the `teaches` property, and they can only teach Courses. Similarly, only Studnets can be `enrolledIn` something, and they can only be enrolled in Courses.

Properties can support inheritance through `rdfs:subPropertyOf:`

```turtle
ex:supervises rdfs:type rdf:Property .
ex:supervises rdfs:subPropertyOf ex:teaches .
```

This means that if a Professor supervises a student, they also teach that student in a broader sense.