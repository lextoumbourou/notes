---
title: Kruskal's Algorithm
date: 2025-02-15 00:00
modified: 2025-02-15 00:00
status: draft
---

**Kruskal's Algorithm** is an algorithm for building a [Minimum Spanning Tree](minimum-spanning-tree.md), an alternative to [Prim's Algorithm](prims-algorithm.md).

* Step 1: Sort all edges in non-decreasing order of their weights
* Step 2: Select the smallest edge that does not form a cycle and add it to the tree
* Step 3: Repeat until all vertices are included
    
## Algorithm

$$
\begin{aligned}
&\textbf{KRUSKAL-MST}(G) \\
&\quad 1. \quad E = \text{edges}(G) \color{purple}{\text{ // get all edges in G}} \\
&\quad 2. \quad \text{sort } E \text{ by weight} \color{purple}{\text{ // sort edges in ascending order}} \\
&\quad 3. \quad T = \text{new Graph}({}, {}) \color{purple}\text{ // create an empty output Graph} \\
&\quad 4. \quad \textbf{for } e \in E \textbf{ do} \color{purple}\text{ // iterate through sorted edges} \\
&\quad \quad\quad \textbf{if } \text{not cycle}(T, e) \textbf{ then} \color{purple}\text{ // check if adding e forms a cycle} \\
&\quad \quad\quad \quad \text{addEdge}(T, e) \color{purple}\text{ // add edge to MST} \\
&\quad \quad\quad \quad \text{addVertex}(T, FROM(e)) \color{purple}\text{ // add the first node} \\
&\quad \quad\quad \quad \text{addVertex}(T, TO(e)) \color{purple}\text{ // add the second node} \\
&\quad 5. \quad \textbf{return } T \color{purple}\text{ // return the final MST}
\end{aligned}
$$


## Code

```python
class Graph:
    def __init__(self, vertices=None, edges=None):
        self.vertices = vertices or set()
        self.edges = edges or []

    def add_vertex(self, vertex):
        self.vertices.add(vertex)
        
    def add_edge(self, from_vertex, to_vertex, weight):
        self.edges.append((from_vertex, to_vertex, weight))

    def find_parent(self, parent, vertex):
        if parent[vertex] == vertex:
            return vertex
        return self.find_parent(parent, parent[vertex])
    
    def union(self, parent, rank, v1, v2):
        root1 = self.find_parent(parent, v1)
        root2 = self.find_parent(parent, v2)
        if rank[root1] < rank[root2]:
            parent[root1] = root2
        elif rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root2] = root1
            rank[root1] += 1

def kruskals_algorithm(G: Graph):
    edges = sorted(G.edges, key=lambda e: e[2])
    T = Graph()
    parent = {v: v for v in G.vertices}
    rank = {v: 0 for v in G.vertices}
    
    for from_v, to_v, weight in edges:
        if G.find_parent(parent, from_v) != G.find_parent(parent, to_v):
            T.add_vertex(from_v)
            T.add_vertex(to_v)
            T.add_edge(from_v, to_v, weight)
            G.union(parent, rank, from_v, to_v)
    
    return T
```



