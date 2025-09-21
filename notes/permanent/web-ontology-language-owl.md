---
title: Web Ontology Language (OWL)
date: 2025-08-23 00:00
modified: 2025-08-23 00:00
status: draft
---

**OWL** or **Web Ontology Language** is an extension to [RDFS](rdfs.md) for creating more expressive and rigorous ontologies on the web. Originally developed by the W3C to extend the capabilities of RDF Schema without replacing it. "OWL" is an intentional acronym joke and stands for Web Ontology Language. I guess OWL is a more memorable acronym than WOL.

OWL distinguishes between two different types of properties:
* **Object Properties**: Connect URL to URL (entity to entity relationships)
* **Data Properties***: Connect URL to literal values (raw data like strings, numbers, dates)

Which allows for constraints and reasoning to be applied based on each time.

Two properties that represent opposite directions of the same relationship:

```turtle
:hasParent owl:inverseOf :hasChild .
:hasSpouse owl:inverseOf :hasSpouse .
```

If John hasChild Mary, then Mary hasParent John.

Equivalent Properties, when properties mean the same thing across ontologies:

```turtle
foaf:knows owl:equivalentProperty rel:friendOf
```

Which enabled data integration across vocabularies.

Has a series of **Property Characteristics**

```turtle
:hasSpouse rdf:type owl:SymmetricProperty .
:hasAncestor rdf:type owl:TransitiveProperty .
:hasSSN rdf:type owl:FunctionalProperty .
```

**Symmetric**: If a related to B, then B related to A.
**Transitive**: If A related to B and C related to C, then A relates to C.
**Functional**: Each entity can have at most one value for this property.

Two URIs can refer to the exact same entity:

```turtle
:JohnSmith owl:sameAs dbpedia:John_Smith .
```

**Disjoint Classes**

Classe that can't overlap. For example, a person can't be a corporation:

```turtle
:Person owl:disjointWith :Organization .
```

OWL allows for defining classes based on property restrictions:

```turtle
:Arranger rdf:type owl:Class ;
    owl:equivalentClass [
        rdf:type owl:Restriction ;
        owl:onProperty :involvedInArrangementActivity ;
        owl:someValuesFrom :ArrangementActivity
    ] .
```

Ttype of restrictions:

* `someValuesFrom` - at least one value must be of the specified type.
* `allValuesFrom` - all values must be of the specified type.
* `hasValue` - must have a specified value.
* `cardinality` - exact number of values.
* `minCardinality/maxCardinality` - min/max number of values.

## Family Example

```turtle
@prefix family: <http://ex.org/family#> .

# Symmetric relationships
family:marriedTo rdf:type owl:SymmetricProperty .
family:siblingOf rdf:type owl:SymmetricProperty .

# Transitive relationships  
family:ancestorOf rdf:type owl:TransitiveProperty .

# Functional properties
family:hasBirthMother rdf:type owl:FunctionalProperty .

# Disjoint classes
family:Male owl:disjointWith family:Female .
```
