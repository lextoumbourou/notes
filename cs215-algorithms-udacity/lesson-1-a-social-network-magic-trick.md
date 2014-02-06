# Lesson 1: A Social Network Magic Trick

* Introduction
    * Will learn: Algorithms that are useful for analysing large social networks
* Magic Trick
    * Collection of actors ("nodes")
    * Connected ("via edges") if they are in a movie together
    * All paths end up at Kevin Bacon in quiz if edges are only traversed once.
        * This became very interesting me after completing the course
    * First course focus:
        * Social networks
            * Connections between individuals that capture info about them
    * 2nd course focus:
        * "magic of algorithms"
            * computation that solves problems
    * Degree of node
        * Number of edges == degree of node
            * 4th degree node = 4 edges
* Eulerian Path
    * Path through the graph that hits every edge exactly once
    * Named after [Leonhard Euler](http://en.wikipedia.org/wiki/Leonhard_Euler)
    * Properties of graphs that have Eulerian Paths
        * All nodes that aren't the start or end node have to have even degree
        * Beginner or end node have to have odd degree (unless it's an Eulerian tour -- see the last point)
        * If the graph is connected and has exactly 2 odd degree nodes, then it has an eulerien path
        * Same if degree of every node in the graph is even
* Importance of mathematics in computer science:
    * Help "think clearly" or formally about what you're doing
    * Analyse correctness of code
    * Analyse efficiency of code
* Naive algorithm
    ```
    def naive(a, b):
        x = a; y = b;
        z = 0
        while x > 0:
            z = z + y
            x = x - 1
        return z
    ```
    * What it does:
        * iterate through the loop a times
            * each time, add y to 0
    * Essentially it just returns the product of a and b 
    * Correctness of algorithm
        * Claim: before or after "while" loop ```ab = xy + z```
        * Base case: first time, x = a, y = b, z = 0, ab = ab + 0 (works)
        * Inductive step: if ```ab = xy + z``` before then ```ab = x'y' + z'``` after
            * Not quite understanding this...
    * Running time of naive(a, b) = Theta(a)
        * Very much linear time
* Russian Peasants algorithm
    ```
    def russian(a, b):
        x = a; y = b;
        z = 0
        while x > 0:
            if x % 2 == 1: z = z + y
            y = y << 1
            x = x >> 1
        return z
    ```
    * Uses bitshift operator
        * Refresher
            ```
            > 17 >> 1
            > 8
            ```
            * Binary for 17 == ```10001```
            * After bit shifting right == ```01000```
    * What it does:
        * When x is odd, add y to z
        * Each iteration (until x is 0)
            * double y
            * halve x until x is 0
    * Example (for intuition)
        * ```russian(4, 5)```
        ```
        | z  | x   | y   |
        | 0  | 4   | 5   |
        | 0  | 2   | 10  |
        | 0  | 1   | 20  |
        | 20 | 0   | 40  | 
        ```
        * Addition occurs only once when x is 1
    * Correctness of algorithm
        * Same strategy for naive applies to russian
* Halving
    * ```floor( log(x, 2) ) + 1```
    * Can work out the answer by trying each formula.
    * Personal: Don't have an intuition for log. Makes it hard to understand. Missing prereqs.
* Divide and Conquer
    * "break a problem into roughly equal sized subproblems, solve separately and combine results"
    * Example for russian algorithm:
        * Multiplication is just repeated addition
        * ```a * b = b + b + b... (a times)```
        * We could divide a times into groups of (a / 2)
    <img src="./divide_and_conquer.png"></img>
