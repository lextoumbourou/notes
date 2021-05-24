# Week 1

## 1.5 Union Find

* About the "Union find problem"

  * Set of algorithms for sovling the "dynamic connectivity" problem. 
  * Look at "Quick find" and "Quick union"

* Steps to developing a usable algorithm -- scientific approach.

  1. Model the problem -- what are the main components of problem that need to be solved?
  2. Find an algorithm to solve it.
  3. Fast enough? Fits in memory?
  4. If not, figure out why.
  5. Find way to address problem.
  6. Iterate until satisfied.


### Dynamic connectivity

* Dynamic connectivity.


  * Given set of N objects:
    * Find / connected query: is there a path connecting any 2 objects?

      <img src="./images/dynamic-connectivity.png"></img>

    * Assumptions:

      * "Is connected to" is an "equivalence relationship:
        * Reflexive: Node A is connected to itself.

            * |Question| Word to describe equivalence relationship when ``p`` is connected to ``p``?

        * Symmetric: If Node A is connected to Node B, Node B is connected to Node A.
          * This might not be the case in a social network graph, for example.

          * |Question| Word to describe equivalence relationship when ``p`` is connected to ``r`` infers ``r`` is connected to ``p``?

        * Transitive: if Node A is connected to Node B, and Node B is connected to Node C, then Node A is connected to Node C.

          * |Question| Word to describe equivalence relationship when ``p`` is connected to ``r`` and ``r`` is connected to ``s``, then ``p`` is connected to ``s``?

    * Connected components

      * "Maximal set of objects that are mutually connected."

        <img src="./images/connected-components.png"></img>

* Implementing the operations

  * Find query - checks if 2 objects are in the same "component" (referring to above section about "Connected components").
  * Union command - union 2 component objects and use to replace the 2 componets when two nodes get connected.

      <img src="./images/union-find-operations.png"></img>

* Union-find data type

  * Goal: make efficient data structure for union-find.
    * Number of objects ``N`` can be huge.
    * Number of operations ``M`` can be huge.
    * Find queries and union commands may be intermixed.
    * Python example:

      ```
      class UnionFind(object):

          def union(p, q):
              """
              Args:
                  p (int) - node 1 to union.
                  q (int) - node 2 to union.
              """

          def connected(p, q):
              """
              Args:
                  p (int) - node 1 to check if connected.
                  q (int) - node 2 to check if connected.

              Return:
                  bool
              """
      ```

### Quick Find

* "Eager" algorithm for solving the connectivity problem.
* Data structure
  * Integer array ``id[]`` of size ``N``.
  * ``p`` and ``q`` are connected if and only if (iff) they have the same id.
  * Objects are denoted by their position in the array. Eg object 0 = pos 0.

    ```
    # obj: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 
    ids = [0, 1, 1, 8, 8, 0, 0, 1, 8, 8]

    # component 1: {0, 5, 6}
    # component 2: {1, 2, 7}
    # component 3: {3, 4, 8, 9}
    ```

  * Find: simply check if p and q have same id.

    ```
    def connected(p, q):
        return ids[p] == ids[q]
    ``` 

  * Union: to merge components containing ``p`` and ``q``, change all entries whose id is ``id[p]`` to ``id[q]``.

    ```
    def union(p, q):
        for val, pos in ids:
            if val == ids[p]:
                ids[pos] == ids[q]
    ```

  * [Python implementation](quick_find.py)
  * [Java implementation](./code/java/QuickFind.class)

  * Run time:

    * Init: ``N``
    * Union: ``N``
    * Find: ``1``

  * Defects:

    * Union too expensive when processing sequence of ``N`` union command on ``N`` objects: quadratic run time (``N**2``)
    * Quadratic time is "unacceptable" because it doesn't scale: eg if the amount of shit you can store in memory increases, the running time will get slower.

### Quick Union

  * "Lazy" approach: don't do work until we have to.
  * Same data structure as Quick Find but different representation in array.
  * Array represents a set of trees called a "forest".
  * Each element points to their parent, parents point to themselves.

  ```
  # 0   1     5
  #          / \
  #         4   2
  #              \
  #               3
  # Obj: 0, 1, 2, 3, 4, 5
  ids = [0, 1, 5, 2, 5, 5]
  ```

  * Find: check if ``p`` and ``q`` have same root.

  ```
  def get_root(i):
      if self.ids[i] != i:
          return get_root(self.ids[i])
      return i

  def find(p, q):
     return self.get_root(p) == self.get_root(q)
  ```

  * Union: Connect the root of ``p`` to the root of ``q``.

  ```
  def union(p, q):
     p_root = self.get_root(p)
     q_root = self.get_root(q)
     self.ids[p_root] = q_root
  ```

  * Run time:

    * Init: ``N``
    * Union: ``N``
    * Find: ``N``  <  worst case

  * Defects:

    * Find too expensive: could be ``N`` array access in case where finding the root needs to interate over all nodes (linear time).

### Quick-Union Improvements

  * "Weighted quick union":

    * Quick union but when combining trees, ensure the smaller tree is linked to larger tree.
    * Have to keep track of size of the tree.
    * Example of union operation in Python.

      ```
      def union(p, q):
          p_root = self.get_root(p)
          q_root = self.get_root(q)
          if p_root == q_root:
              return

          if self.size[p] < self.size[q]:
              self.size[p_root] = q_root
          else:
              self.size[q_root] = p_root
      ```
    * Run times:

      * Init: ``N``
      * Find: ``log N`` - proportional to depth of ``p`` and ``q``.
      * Union: ``1`` - constant time, given roots.

    * Depth of node ``x`` is at most ``log(N, base=2)``

      * Increase by 1 when tree ``T1`` containing x is merged with ``T2``.
      * Size of tree containing ``x`` at least doubles since ``|T2| >= |T1|``.
      * Size of tree containing ``x`` can double at most ``log(N, 2)`` times.
      * |Question| The depth of any node is at most what?

  * "Path compression"

    * After computing root of ``p``, we point every node in the tree to the root.
    * Running time is roughly linear (log * N).
    * No such algorithm for Union Find that's exactly linear, but this is close enough.

### Applications

  * Percolation
    * Model for many physical systems
    * Each site is open with probability `p`` or blocked with probability ``1 - p``.
    * System "percolates" iff top and bottom are connected by open sites.

      <img src="./images/likelihood-of-percolation.png"></img>

    * Phase transition: when ``p`` is less than some value the space will almost certainly perculate and vice versa.
      * No solution to problem; need to use simulations to find solution. Fast union find is required.
      * Threshold is roughly: 0.592746

## 1.4 Analysis of Algorithms

  * Reason to analyze algorithms

    * Predict performance.
    * Compare algorithms.
    * Provide guarantees.
    * Understand theoretical basis.
    * Practical reason: avoid performance bugs.

  * Algorithmic "successes"

    * Discrete Fourier transform:

      * Break down waveform of N samples into "periodic components".
      * Basis for DVDs, JPEGs.
      * Brute force: ``N**2`` steps.

  * "Scientific method" 

    * *Observe* a feature of the natural world.
    * *Hypothesize* a model that is consistent with observations.
    * *Predict* events using hypothesis.
    * *Verify* predictions by making further observations.
    * *Validate* by repeating until the model hypothesis and observations agree.

  * Principles:

    * Experiments must be *reproducible*.
    * Hypothese must be *falsifiable*.

  * Question notes:

    * ``lg`` = base-2 logarithm function
      * ``lg N`` == ``log(N, 2)`` == "what exponent do you need to multiply 2 to get to N?" == What is ``i`` in ``2 ** i == N``?
      * |Question| Describe what ``log(N, 2)`` is asking in a sentence.

  * Observations

    * "Make observations about running time of program."
    * Example: 3-Sum

      * "Given ``N`` distinct integers, how many triple sum to exactly zero?

         ```
         ints = [30, -40, -20, -10, 40, 0, 10, 5]
         get_triples_count(N)

         # sum([30, -40, 10]) == 0
         # sum([30, -20, -10]) == 0
         # sum([-40, 40, 0]) == 0
         # sum([-10, 0, 10]) == 0
         # Expected result == 4
         ```

      * Brute force run time == ``N**3``
        * "for each item, go through each item, then go through each item for each of that item."
        * [C example](brute_force_3_sum.c)

    * Measuring run time of program

      * Empirical analysis: run it for different sizes and measure how long it takes.
      * Standard plot:

        * Plot running time T(N) vs input size N using log-log scale.
        * Use Regression to fit straight line through data point:

          <img src="./images/log-log-run-time-plot.png"></img>

        * Can then use to figure out running time for different values of ``N``.

        * Power law notes:

          *  Relationship between two quantites: change one, it changes the other proportionally.

      * Doubling hypothesis:

        * Double the size of input and take the ratio.

    * Experimental algorithmics:

      * System independent effects:

        * Algorithm
        * Input data.

      * System dependent effects:

        * Hardware
        * Software (compiler, interpreter, garbage collecter)
        * System (os, network etc)

  * Mathematical Models

    * Total running time: sum of cost * frequency for all operations.
    * Simplications made when calculating running time:

      * 1. Cost model: Look at operation execute most frequently or that's most expensive (eg array access in 2-sum problem)
      * 2. Tilde notation: ignore low order terms
        * When ``N`` is large, terms are negligible.
        * When ``N`` is small, who cares?
        * Technical definition: ``f(N) ~ g(N)`` == "Limit of ``f(N) / g(N) == 1`` as N approaches infinity"

    * Estimating a discrete sum:

      * Don't know calculus: totally lost. :)
