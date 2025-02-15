---
title: Prim's Algorithm
date: 2025-02-15 00:00
modified: 2025-02-15 00:00
status: draft
---

Prim's Algorithm is an algorithm for building a [Minimum Spanning Tree](../../../permanent/minimum-spanning-tree.md) by:

1. Starting with a randomly selected node.
2. Repeatedly adding the lowest weight edge that connects the current tree to a new node.
3. Continuing until all nodes are included.

* The main steps of the algorithm are:
    * Step 1: Initialise the tree with any vertex from the graph
    * Step 2: Incrementally construct the tree by finding and adding minimum weight edges and their connected nodes
    
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

## Visualisation

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <defs>
    <!-- Animations -->
    <animate id="fadeIn" attributeName="opacity" from="0" to="1" dur="0.5s" fill="freeze"/>
    
    <!-- Node highlight -->
    <animate id="nodeHighlight" attributeName="fill" 
             values="#ccc;#98fb98;#98fb98" 
             dur="1s" 
             fill="freeze"/>
             
    <!-- Edge highlight -->
    <animate id="edgeHighlight" attributeName="stroke" 
             values="#666;#ff0000;#ff0000" 
             dur="1s" 
             fill="freeze"/>
  </defs>

  <!-- Background -->
  <rect width="400" height="300" fill="#fff"/>
  
  <!-- Edges -->
  <g stroke-width="2">
    <line x1="100" y1="150" x2="200" y2="80" stroke="#666">
      <title>Weight: 4</title>
      <!-- A to B -->
      <animate attributeName="stroke" values="#666;#ff0000" dur="3s" begin="3s" fill="freeze"/>
    </line>
    
    <line x1="200" y1="80" x2="300" y2="150" stroke="#666">
      <title>Weight: 2</title>
      <!-- B to C -->
      <animate attributeName="stroke" values="#666;#ff0000" dur="3s" begin="2s" fill="freeze"/>
    </line>
    
    <line x1="300" y1="150" x2="200" y2="220" stroke="#666">
      <title>Weight: 1</title>
      <!-- C to D -->
      <animate attributeName="stroke" values="#666;#ff0000" dur="3s" begin="1s" fill="freeze"/>
    </line>
    
    <line x1="100" y1="150" x2="300" y2="150" stroke="#666">
      <title>Weight: 3</title>
      <!-- A to C -->
      <animate attributeName="stroke" values="#666;#ff0000" dur="3s" begin="0s" fill="freeze"/>
    </line>
    
    <line x1="100" y1="150" x2="200" y2="220" stroke="#666">
      <title>Weight: 6</title>
      <!-- A to D -->
    </line>
  </g>

  <!-- Nodes -->
  <g>
    <!-- Node A -->
    <circle cx="100" cy="150" r="20" fill="#ccc" stroke="black" stroke-width="2">
      <animate attributeName="fill" values="#ccc;#98fb98" dur="1s" begin="0s" fill="freeze"/>
    </circle>
    <text x="100" y="155" text-anchor="middle" font-family="Arial" font-size="16">A</text>
    
    <!-- Node B -->
    <circle cx="200" cy="80" r="20" fill="#ccc" stroke="black" stroke-width="2">
      <animate attributeName="fill" values="#ccc;#98fb98" dur="1s" begin="3s" fill="freeze"/>
    </circle>
    <text x="200" y="85" text-anchor="middle" font-family="Arial" font-size="16">B</text>
    
    <!-- Node C -->
    <circle cx="300" cy="150" r="20" fill="#ccc" stroke="black" stroke-width="2">
      <animate attributeName="fill" values="#ccc;#98fb98" dur="1s" begin="1s" fill="freeze"/>
    </circle>
    <text x="300" y="155" text-anchor="middle" font-family="Arial" font-size="16">C</text>
    
    <!-- Node D -->
    <circle cx="200" cy="220" r="20" fill="#ccc" stroke="black" stroke-width="2">
      <animate attributeName="fill" values="#ccc;#98fb98" dur="1s" begin="2s" fill="freeze"/>
    </circle>
    <text x="200" y="225" text-anchor="middle" font-family="Arial" font-size="16">D</text>
  </g>

  <!-- Edge Weights -->
  <g font-family="Arial" font-size="14">
    <text x="150" y="100" text-anchor="middle">4</text>
    <text x="250" y="100" text-anchor="middle">2</text>
    <text x="250" y="200" text-anchor="middle">1</text>
    <text x="200" y="140" text-anchor="middle">3</text>
    <text x="150" y="200" text-anchor="middle">6</text>
  </g>
</svg>
