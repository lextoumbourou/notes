## Lesson 5: Weighed graphs

* Begins with a "Magic Trick" about cutting a 6-pack ring thing so that a turtle's neck can't get caught in it.
    * The point? Not sure.
* Strength of connections
    * A decent thought experiment. I brute forced the way through the result.

    ```
    highest = (None, 0)
    for char in strength.keys():
        for comic in graph[char].keys():
            for linked_char in graph[comic]:
                if char != linked_char:
                    strength[char][linked_char] = strength[char].get(linked_char, 0) + 1
                    if strength[char][linked_char] > highest[1]:
                        highest = ((char, linked_char), strength[char][linked_char])
    
    print highest
    ```

    * The solution put the results into a heap to get the top K results
* Weighed Social Networks
    * other measures of strength of connections (aside from how many comics book you've been in together)
        * how often they email each other
        * how long they've known each other
        * frequency of meetings
        * how often they rate each other's posts
        * # of news articles both have read
        * how often they text each other
* shortest weighed paths - "cheaps roots from one person to another"
* rare moment that was actually about "connected social networks"

![](http://i.imgur.com/bAZzs0X.jpg) 
 
* Simulating the algorithm is in page 2 of my notebook.

## PS 5

* Question 2:
	* Need to understand Dijkstra's Shortest Path algorithm. I wrote it out by hand a couple of times until I get it.
	* Need to store the path alongside the cost. Maybe use a tuple of a dictionary? 
