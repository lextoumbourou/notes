# Lesson 3: Basic Graph Algorithms

* Degrees and paths are not that well covered.
    * What's a degree?
    * What's a path?
    * Is this prerequisite knowledge?
    * Degree appears to be number of edges that touch one node.
    * "Graph exhibits the small world phenmonemnon if nodes have relativiely small degree but also if they have a small path to other arbitrary nodes."
* Clique
    * Each node is connected to each other nodes
    * Degree
        * linear
        * Theta(n)
    * Path
        * Constant
        * Theta(1)
* Ring
    * Degree
        * All nodes have degree of 2 
        * Constant
        * Theta(1)
    * Path
        * Linear (what?)
* Balanced tree
    *  Degree
        * Constant
        * There are never more than 3 edges per node
        * Theta(1)
    * Path
        * To get from one edge to the other, the lowest child just has to get to the root node (which is log(n)) and back down
        * 2 log n
        * Theta(log n)
* Hypercube
    * Each node is numbered with log n bits
    * Degree
        * Theta(log n)
    * Path
        * To go from any node in a hypercube to any other node, it's at most log n paths
        * Theta(log n) 
* Clustering coeffiecient
    * Cliquishness: how likely is it that two nodes that are connected are part of some larger groups of highely connected nodes
    * v: a node
    * Kv: its degree
    * Nv: number of links between neighbours of V
    ```cc(v) = (2 * Nv) / (Kv * (Kv - 1))```
            ![](http://i.imgur.com/lgSpMrs.jpg)
    * We don't add all the neighbours nodes together, we just pick one?
