* Eulerian Path
    * 

---

What is 17 >> 1 ?
17 == 10001
17 >> 1 == 01000

>>> 0b10001 >> 1
8

---

def russian(a, b):
     x = a; y = b
     z = 0
     while x > 0:
          if x % 2 == 1:
               z = z + y
          y = y << 1
          x = x >> 1
     return z

russian(4, 2)

* x = 5; y = 2; z = 0

y    x    z
4    2   2
8    1   2
16  0   10

return 10

---

- Confirm correctness of Russian with formula
     ab = xy + z

--

How many additions for russian(20, 7)

x     y     z   adds

20   7     0
10   14   0
5     28   0
2     56   28  1
1     112  28  
0     224 140 2

2

----

Eulerian path rules

1. All nodes that are not beginning and ending has to have even degree. In this case, beginning and end will have odd degree.
2. If all nodes in the graph is even (and the graph is connected) then it has an eulerian path.

Hierholzer's algorithm looks like this:

1. If the graph is Eulerian (all nodes even degree), start from any vertex. If semi, start from the odd vertex.
2. When edge is visited, add vertices to the path and remove edge from the graph so it won't be visited again.
3. If graph is semi-eulerian, you might come to other odd vertex without traversing entire path. If eulerian, might come to starting vertex without traversing the entire graph. Here, unvisited edges make up Eulerian graph.
4. In 3rd case, look for vertices already in the graph with non-visited edges. Start traversing back until you find one with edge still in graph. Traverse through graph until you find same edge. Add the rest of the path back.

-
