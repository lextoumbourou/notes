Title: Understanding Dijkstra's Shortest Path Algorithm
Tagline: How to find your way out of a maze like a robot
Date: 2014-01-12
Tags: Computer Science, Graphs
Status: Draft

</p>

<script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/fabric.js/1.4.0/fabric.min.js"></script>

## Overview

Dijkstra's Shortest Path Algorithm provides a simple way for a computer to build a shortest path tree for a graph with non-negative path costs. Don't worry if you don't know what the means yet, I'm going to explain the algorithm initially using no code, with the initution that helped me understand it. I hope that it does th esame for you.

<div class="img-annotated">
    <img src="/images/hardware_robot.jpg" alt="Robot from movie Hardware (1990)">
</div>

## The scene

A robot finds itself in a maze. The robot has no map of the maze. All it knows is that in each corner of the maze there are checkpoints, which we'll call **nodes**.  There are paths between checkpoints, which we'll call **edges**. Each edge has a cost, literally a dollar amount that it needs to pay to traverse the edge. None of the edges have negative path costs, in other words, the robot can't earn money taking any of the edges.

At each node, the robot can ask for information about the node's neighbours, including price to traverse the edge.

The maze looks like this:

<canvas id="example1">

    (A) <--->  (B)
     ^          ^ 
     |    |     |
     |    |     |
     |    |     |
     |    |     |
     ^          ^
     (C) <---> (D)

</canvas>

<script>
    var canvas = new fabric.Canvas('example1');
    var nodeA = new fabric.Text('A', {
        left: 5,
        top: 5,
        shadow: 'rgba(0,0,0,0.3) 5px 5px 5px',
        fontSize: 20});
    canvas.add(nodeA);
    var nodeB = new fabric.Text('B', {
        left: 200,
        top: 5,
        shadow: 'rgba(0,0,0,0.3) 5px 5px 5px',
        fontSize: 20});
    canvas.add(nodeB);
    var nodeC = new fabric.Text('C', {
        left: 5,
        top: 100,
        shadow: 'rgba(0,0,0,0.3) 5px 5px 5px',
        fontSize: 20});
    canvas.add(nodeC);
    var nodeD = new fabric.Text('D', {
        left: 200,
        top: 100,
        shadow: 'rgba(0,0,0,0.3) 5px 5px 5px',
        fontSize: 20});
    canvas.add(nodeD);

    var pathA2pathB = new fabric.Rect({
        left: 30,
        top: 20,
        width: 170,
        height: 1,
        fill: 'black',
    });
    canvas.add(pathA2pathB);
</script>

Our robot begins at ```Node A```.

The robot is assigned to find the cheapest way to navigate the maze, from checkpoint to checkpoint. The robot can only store two tables of information:

* The first is called the **Frontier**. This is where it stores information about what nodes it can travel to next.
* The second is the **Explored** list. Once a node makes it to this list, the robot will never explore it again.
 
1\. The robot begins at node A. It adds it to the *Explored* list. Since we haven't spent a cent, the cost to this path is 0.

    > Explored list <

    Node   Cost   Path
    A      0      A

> Robot: "Node A, what are your neighbours?"

> Node A: "There's B, with a cost of $5 and C with a cost of $3."

Since neither of these nodes is on the **Explored** list, it puts them both on the **Frontier**. The robot adds the cost spent so far (current $0) to the price to get to these nodes.

    > Frontier list <

    Node   Cost           Path
    B      $0 + $5 = $5   A -> B
    C      $0 + $3 = $3   A -> C

2\. The Robot decides where to travel next. It searches the **Frontier** list for the cheapest path. It finds C.

3\. Now at C, the robot moves it from the **Frontier** to the **Explored** list. It is impossible for the robot to find a cheaper path to C, because if such a path existed, it would have found it already.

    > Explored list <

    Node    Cost    Path 
    A       0       A
    C       $3      A -> C

    > Frontier list <

    Node    Cost       Path
    B       $5         A -> B

The robot asks the node for its neighbours.

> Node C: "My neighbours are nodes A, E and B. A costs you $3 dollars to return, E costs you $2 and B will cost you $1."

The Robot knows ```A``` is on the **Explored** list, so it doesn't need to explore that again. The robot hasn't explored ```E```, so it adds the **Frontier**, with the path and new cost. The robot goes to add ```B``` to the **Frontier**, but realises that it's already on it. However, the cost of path to B through ```A -> C -> B``` is only $4 ($0 + $3 + $1) which is less than the path it has stored. So, the robot removes the old path to B and adds the new.

    > Frontier list <
    
    Node    Cost            Path
    B       $3 + $1 = $4    A -> C -> B    
    E       $3 + $2 = $5    A -> C -> E

The robot scans the **Frontier** list for the next destination. Since B is the cheapest path so far, it follows it.

4\. Now at B, the Robot takes B from the Frontier and adds it to Explored.

    > Explored list <

    Node    Cost            Path 
    A       0               A
    C       $3              A -> C
    B       $3 + $1 = $4    A -> B -> B    

    > Frontier list < 

    Node    Cost            Path
    D       $3 + $6 = $9    A -> C -> D

> Robot: "Excuse me Node B, please tell me about your neighbours."

> Node B: "Well Mr Robot. There's A, which will cost you $5 to visit."

> Robot: "I know about A. What else you got?"

> Node B: "Okay, there's C - "

> Robot: "Yes, C's where I just came from. Anything else?"

> Node B:  Well, then there's D for a cost of $2"

The Robot checks the Frontier. It's got a path to D, but the path to B + the path to D is cheaper than what's on the Frontier. So, it replaces the D path with the new one

    > Frontier list <
    
    Node     Cost            Path
    D        $4 + $2 = $6    A -> B -> D

The robot only has one node on the Frontier. It travels to D.

5\.  The Robot adds D to the Explored list

    > Explored list <

    Node    Cost            Path
    A       0               A
    C       $3              A -> C
    B       $3 + $1 = $4    A -> C -> B
    D       $4 + $2 = $6    A -> B -> D

> Robot: "D, tell you your neighbours."

> Node D: "I have C and B."

Both visited.

The robot checks its Frontier list. Empty.

The job is done. Queue the soundtrack.

<iframe width="560" height="315" src="//www.youtube.com/embed/YbmqfSo_zq4" frameborder="0" allowfullscreen></iframe>

## In code

So, now let's see it as Python code. For the remainder of the blog post, the code with get built in the right-hand side as you scroll down.

Firstly, the way we can represent a graph like the maze is something like this.

    maze_graph = {
        'A': {'B':5, 'C': 6},
        'B': {'A':5, 'D': 2},
        'C': {'A':5, 'D': 2},
        'D': {'B': 2,}

You can see that A is connected to B with a cost of $5, and if you look at B, it's connected to A with the same cost. Because each node is connected back to each other, we call the graph "Undirected".

I'm going to create the ```frontier``` list as a dictionary, where the Node is the key. Each value will be a tuple, where the first value is the cost, the second the path (represented as a list of nodes). Like this:

    frontier = {node: (cost, path)}

Since we're starting at A, I'll add it to the frontier dictionary with a cost of $0.

    frontier['A'] = {
        (0, ['A'])
    }

Tthe explorer list will be another dictionary, in the same format as the ```frontier```.

    explored = {}

Before we go on, I'm going to define a function called ```get_smallest_node(frontier)```. This implementation of function is quite important, it's the different between this being a "fast" algorithm and a "slow" one. In my implementation, I will loop through all the nodes on the frontier each time I visit a node. A better way to do it is to use a data structure that allows for quickly finding the minimum like a "heap queue", but that's out of scope of this article for now. The function looks something like this:

    def get_smallest_node(frontier):
        # Start off by setting the smallest value to infinity
        smallest_node, smallest_value = None, float('inf')

        # Go through the frontier, each time we find a smaller value than 
        # what we've stored, we save it
        for node in frontier:
            cost = frontier[node][0]
            if cost < smallest_value:
                smallest_value = cost
                smallest_node = node

        return smallest_node

While there is nodes to visit on the Frontier, we're going to follow the procedure

    while frontier:

Firstly, we call the ```get_smallest_node``` function, the robot then visits the node and adds it to the ```explored``` list, removing it from the ```frontier```.

        node = get_smallest_node(frontier)
        explored[node] = frontier.pop(node)

Next, we ask the node for each of its neighbours

        for neighbour in maze[node]:
        

Firstly, if the node is in ```explored``` we ignore it. 

            if neighbour in explored:
                continue

If the neighbour is not in the ```frontier```, we add it. When we add it, we make the cost, the cost from node -> neighbour + the cost so far to node. We also make the add the neighbour to the path so far. Like so:

            if neighbour not in frontier:
                 cost_so_far = explored[node][0]
                 cost_to_neighbour = maze[node][neighbour][0]
                 total_path_cost = cost_so_far + cost_to_neighbour
                 new_path = explored[node][1] + [neighbour]
                 frontier[neighbour] = (
                     total_path_cost, new_path)

One other thing we need to do, is update the ```frontier``` if we find a path that's *cheaper* than the one we already have, we can do that by adding an extra statement to the ```neighbour not in frontier``` check which also updates the ```frontier``` if the cost we get is less than the cost we already have in the frontier:

            if neighbour not in frontier or (total_path_cost < frontier[node][0]):
                 cost_so_far = explored[node][0]
                 cost_to_neighbour = maze[node][neighbour][0]
                 total_path_cost = cost_so_far + cost_to_neighbour
                 new_path = explored[node][1] + [neighbour]
                 frontier[neighbour] = (
                     total_path_cost, new_path)


Let's see all that code together:


    def get_smallest_node(frontier):
        # Start off by setting the smallest value to infinity
        smallest_node, smallest_value = None, float('inf')

        # Go through the frontier, each time we find a smaller value than 
        # what we've stored, we save it
        for node in frontier:
            cost = frontier[node][0]
            if cost < smallest_value:
                smallest_value = cost
                smallest_node = node

        return smallest_node

	 maze_graph = {
	     'A': {'B':5, 'C': 6},
	     'B': {'A':5, 'D': 2},
	     'C': {'A':5, 'D': 2},
	     'D': {'B': 2}
     }

    frontier = {
       'A': (0, ['A'])
    }
    explored = {}

    while frontier:
        node = get_smallest_node(frontier)
        explored[node] = frontier.pop(node)

        for neighbour in maze[node]:
            if neighbour in explored:
                continue

            total_path_cost = explored[node][0] + maze[node][neighbour][0]
            if (neighbour not in frontier) or \
               (total_path_cost < frontier[neighbour][0]):
	                 new_path = explored[node][1] + [neighbour]
	                 frontier[neighbour] = (
	                     total_path_cost, new_path)

And, that's it. I hope that helps you wrap your head around Dijkstra's Shortest Path Algorithm.
