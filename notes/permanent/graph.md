---
title: Graph
date: 2023-04-09 00:00
modified: 2025-02-22 00:00
cover: _media/konigsberg-bridges.png
summary: the study of graphs
hide_cover_in_article: true
aliases:
- Graphs
tags:
- ComputerScience
- GraphTheory
---

A [Graph Theory](graph-theory.md) is a visual representation of interconnected systems using circles and lines. Circles represent nodes or vertices. Lines represent links or edges. Graphs are used to analyze and solve problems in various interconnected systems.

```mermaid
graph LR
A((Node A)) --- B((Node B))
A --- C((Node C))
B --- D((Node D))
C --- D
B --- C

classDef node1 fill:#b9b28b,stroke:#1B3D2F,stroke-width:2px;
classDef node2 fill:#8ba7b9,stroke:#1B3D2F,stroke-width:2px;
classDef node3 fill:#b98b99,stroke:#1B3D2F,stroke-width:2px;
classDef node4 fill:#7A976B,stroke:#1B3D2F,stroke-width:2px;

class A node1;
class B node2;
class C node3;
class D node4;
```

## Historical Origin: The [Seven Bridges of Königsberg](seven-bridges-of-knigsberg.md)

![konigsberg-bridges.png](../_media/konigsberg-bridges.png)

Graph theory originated from a famous mathematical problem in the 18th century in Königsberg (now Kaliningrad, Russia). The city had seven bridges connecting four land areas separated by the Pregel River. The puzzle asked whether it was possible to walk through the city crossing each bridge exactly once.

In 1735, mathematician Leonhard Euler proved this was impossible by abstracting the problem. He represented land masses as points (A, B, C, D) and bridges as lines connecting them, creating what would become the foundation of graph theory.

```mermaid
graph TD
A((Land A)) --- B((Land B))
A --- B
A --- C((Land C))
A --- D((Land D))
B --- D
C --- D

classDef landmass fill:#8ba7b9,stroke:#1B3D2F,stroke-width:2px;

class A,B,C,D landmass;
```

Euler's insight was that for a complete path to exist (crossing each bridge exactly once), each land mass except for the starting and ending points must have an even number of bridges connected to it. Since all four land areas in Königsberg had an odd number of bridges, such a path was impossible.

## Graph Classifications

Graphs can be classified in several ways based on their properties:

### Directed vs. Undirected

#### [Directed Graphs](directed-graphs.md)

In directed graphs, edges have a specific direction indicated by arrows. These represent one-way relationships.

**Examples**: Twitter following relationships, one-way streets in a city map, workflow diagrams

```mermaid
graph LR
A((User A)) -->|follows| B((User B))
B -->|follows| C((User C))
A -->|follows| C
D((User D)) -->|follows| A

classDef user fill:#b9b28b,stroke:#1B3D2F,stroke-width:2px;
class A,B,C,D user;
```

#### [Undirected Graphs](undirected-graphs.md)

In undirected graphs, edges have no direction and represent symmetrical relationships.

**Examples**: Facebook friendships, telecommunication networks, chemical bonds

```mermaid
graph LR
A((Person A)) --- B((Person B))
B --- C((Person C))
C --- D((Person D))
A --- D
A --- C

classDef person fill:#8ba7b9,stroke:#1B3D2F,stroke-width:2px;
class A,B,C,D person;
```

### Weighted vs. Unweighted

#### [Weighted Graphs](weighted-graphs.md)

In weighted graphs, edges have different values or importance attached to them.

**Examples**: Road networks where weights represent distances, communication networks where weights represent bandwidth

```mermaid
graph LR
A((City A)) -->|120 km| B((City B))
B -->|85 km| C((City C))
A -->|200 km| C
C -->|60 km| D((City D))

classDef city fill:#b98b99,stroke:#1B3D2F,stroke-width:2px;
class A,B,C,D city;
```

#### Unweighted Graphs
In unweighted graphs, all edges have equal importance.

**Examples**: Simple connection diagrams, logical relationships

## Common Graph Topologies

Graphs come in various standard topologies, each with specific use cases:

### Bus Topology

Nodes are arranged in a line, with each node connected to adjacent nodes.

```mermaid
graph LR
A((Node A)) --- B((Node B)) --- C((Node C)) --- D((Node D)) --- E((Node E))

classDef busNode fill:#7A976B,stroke:#1B3D2F,stroke-width:2px;
class A,B,C,D,E busNode;
```

### Ring Topology

Similar to a bus, but the last node connects back to the first, forming a circle.

```mermaid
graph LR
A((Node A)) --- B((Node B))
B --- C((Node C))
C --- D((Node D))
D --- E((Node E))
E --- A

classDef ringNode fill:#b9b28b,stroke:#1B3D2F,stroke-width:2px;
class A,B,C,D,E ringNode;
```

### Tree Topology

Follows a hierarchical tree data structure, with a root node and branches.

```mermaid
graph TD
A((Root)) --- B((Node B))
A --- C((Node C))
B --- D((Node D))
B --- E((Node E))
C --- F((Node F))

classDef treeNode fill:#8ba7b9,stroke:#1B3D2F,stroke-width:2px;
class A,B,C,D,E,F treeNode;
```

### Regular Manhattan Topology

A grid-like structure resembling city blocks.

```mermaid
graph TD
A((Node A)) --- B((Node B)) --- C((Node C))
A --- D((Node D)) --- E((Node E)) --- C
D --- F((Node F)) --- G((Node G)) --- E

classDef gridNode fill:#b98b99,stroke:#1B3D2F,stroke-width:2px;
class A,B,C,D,E,F,G gridNode;
```

### Arbitrary Mesh Topology

Random interconnected pattern with no specific structure.

```mermaid
graph TD
A((Node A)) --- B((Node B))
A --- C((Node C))
A --- E((Node E))
B --- D((Node D))
C --- D
C --- F((Node F))
D --- E
E --- F

classDef meshNode fill:#7A976B,stroke:#1B3D2F,stroke-width:2px;
class A,B,C,D,E,F meshNode;
```

## Real-World Applications

Graphs have countless real-world applications across different domains:

### Literature Analysis

In literature analysis, graphs can represent character interactions. For example, in "Les Misérables," nodes represent characters and edges show interactions between them, revealing the structure of relationships within the novel.

### Movie Plot Mapping

Similar to books, graphs can be used to visualize character interactions in films. For instance, in "The Imitation Game," a graph could show relationships between Alan Turing and other characters, highlighting the central role of certain individuals.

### Computer Networks

In computer networks, nodes represent routers or computers, while edges represent communication links. The 1991 NSFNET (National Science Foundation Network) backbone can be visualized as a graph showing the early structure of what would become the internet.

### Social Networks

Social media platforms use graph theory extensively. Nodes represent people, and edges indicate friendships or following relationships. This helps identify communities, influential users, and information flow patterns.

```mermaid
graph TD
subgraph "Community A"
A((User A)) --- B((User B))
A --- C((User C))
B --- C
end

subgraph "Community B"
D((User D)) --- E((User E))
D --- F((User F))
E --- F
end

C --- D

classDef communityA fill:#b9b28b,stroke:#1B3D2F,stroke-width:2px;
classDef communityB fill:#8ba7b9,stroke:#1B3D2F,stroke-width:2px;

class A,B,C communityA;
class D,E,F communityB;
```

Graph theory provides a powerful framework for understanding complex systems across disciplines, from mathematics and computer science to sociology and literature.