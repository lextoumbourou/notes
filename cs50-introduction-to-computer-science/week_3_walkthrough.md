## Comment generate.c
* generate uses a pseudo-random number generator (rand())
* same seed? same sequence of numbers

## Piping
```prog > file.txt
prog < file.txt
prog1 | prod2```

# Makefile
* specify what happens when you make something

## Bubble sort pseudo code
while elements have been swapped
    swapped = false
    for i = 0 to n - 2
        if array[i] > array[i + i]
            swap array[i] and array[i + 1]
            swapped = true

## Selection sort
for i = 0 to n -1
min = i
    for j = i + 1
        if array[j] < array[min]
            min[j]
    if array[min] != array[i]
        swap array[min] and array[i]

## Search
* Linear search does not require a sorted array, as it just goes through all 1 at a time (hence O(n))
* Binary search, on the other hand, does and it's O(log n)

### Binary search pseudo code
```while length of list > 0
    look at middle of list
    if number found, return true
    else if number too high, only consider left half
    else if number too low, only consider right half
    return false```

## Fifteen
### init()
* board needs to contain starting state of board
    * board[x][y] could contain elements at (x, y)
    * board[x][y] could contain element at row x and column y
* board starts off in descending order
    * condition to make board solvable: if number of tiles odd in a row, swap 1 and two
* keep track of blank title
    * use a constant value to represent blank space

### draw()
* output current state of board to terminal
* print new line at the end of the row
* print spaces between columns
* print using leading space

### move()
* Allows the user to move tiles around on board
* To move a tile, change the board array
* Move requires you to search for board array to find the number the user specified.
* Once we move a tile, we should find where the blank space is and cache it.
* If positions are adjacent, above, below, left, right are only legal moves.
* Can't move above max or below 0, easy! 

### won()
* need to iterate over entire board array, if one thing is out of order, then the game hasn't been one.
* if one tile is wrong, the whole game is wrong.
