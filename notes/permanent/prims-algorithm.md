---
title: Prim's Algorithm
date: 2025-02-15 00:00
modified: 2025-02-15 00:00
status: draft
---

**Prim's Algorithm** is an algorithm for building a [Minimum Spanning Tree](minimum-spanning-tree.md), alternative to [Kruskal's Algorithm](kruskals-algorithm.md).

1. Starting with a randomly selected node.
2. Repeatedly adding the lowest weight edge that connects the current tree to a new node.
3. Continuing until all nodes are included.

## Algorithm

$$
\begin{aligned}
&\textbf{PRIM-MST}(G) \\
&\quad 1. \quad vs = vertices(G) \color{purple}{\text{ // get all vertices in G}} \\
&\quad 2. \quad T = \text{new Graph(FIRST}(vs), {}) \color{purple}\text{ // create a new output Graph which starts with the first element in G} \\
&\quad 3. \quad \textbf{while } (|T| < |G|) \textbf{ do} \color{purple}\text{ // repeat until all nodes are in output Graph} \\
&\quad \quad\quad \color{purple}\text{  } \\
&\quad \quad\quad \color{purple}\text{  // get all possible edges to nodes not in the graph} \\
&\quad 4. \quad\quad L = {e \mid e \in edges(G) \text{ and FROM}(e) \in T \text{ and TO}(e) \in G \text{ and TO}(e) \notin T} \\
&\quad \quad\quad \color{purple}\text{  } \\
&\quad 5. \quad\quad newE = \min(weight(e)) \text{ in } L \color{purple}\text{  // get the lowest code edge} \\
&\quad 6. \quad\quad \text{addVertex}(T, \text{TO}(e)) \color{purple}\text{  // add the connected node to new edge} \\
&\quad 7. \quad\quad \text{addEdge}(T, newE) \color{purple}\text{  // add the edge to the graph}
\end{aligned}
$$


## Code

```python
class Graph:
    def __init__(self, vertices=None, edges=None):
        self.vertices = vertices or set()
        self.edges = edges or {}

    def __len__(self):
        return len(self.vertices)
    
    def add_vertex(self, vertex):
        self.vertices.add(vertex)
        
    def add_edge(self, from_vertex, to_vertex, weight):
        self.edges[(from_vertex, to_vertex)] = weight

def prims_algorithm(G: Graph):
    vs = G.vertices
    T = Graph(first(vs), {})
    while len(T) < len(G):
        # Create set of possible edges
        L = {}
        for (from_v, to_v), weight in G.edges.items():
            if (from_v in T.vertices and to_v in G.vertices and to_v not in T.vertices):
                L[(from_v, to_v)] = weight
            elif (to_v in T.vertices and from_v in G.vertices and 
                  from_v not in T.vertices):
                L[(to_v, from_v)] = weight
        if not L:
            break
            
        new_edge = min(L.items(), key=lambda x: x[1])
        from_vertex, to_vertex = new_edge[0]
        weight = new_edge[1]
        
        T.add_vertex(to_vertex)
        T.add_edge(from_vertex, to_vertex, weight)
        
    return T
```