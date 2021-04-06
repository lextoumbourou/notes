case-switch 
==========
* Can only use integers

structs
=======
* allows you to group variables into a single structure
    * g is a global struct containing game informatino
    * variables can be of different types
* g.y, g.x row and column of cursor
* g.board: 2D array representing board
* g.top, g.left: coordinates of top-left point of board
* g.number: number of board

Functions
========
* restart_game()
    * start a new game with the board specified in g.board
* draw_borders()
* draw_grid()
* draw_logo()
* draw_numbers()
* show_cursor()
    * set the position of the cursor based on g.y and g.x
* show_banner(char* b)
    * show the string b as a banner
* hide_banner()
    * hide the currently-shown banner

```pidof sudoku```
start gdb with:
```gdb sudoku pid```

ncurses
=======
* write GUI terminal apps
* operates row, column (y, x)
* functions:
    ```move()```
    * Move the cursor to the given (y, x) location
    ```mvaddstr(int y, int x, char c)```
    * Move to (y, x) then print c there.
    ```getch()```
    * Get a single character from the user (returns a char)
    * KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT represent a few keys
        * #defined constants by ncurses
        * CTRL('l') represents Ctrl+l

TODO
====
* move cursor
* input number
* move legality
* input blank
* check if won

show_cursor()
============
* show_cursor() moves the cursor based on g.y and g.x
    * different g.y or g.x == different cursor location
    * don't worry about move or mvaddch

Moving
======
* increment/decrement g.y or g.x based on arrow pressed
    * main only handles 'N', 'R' and CTRL('l')
    * (0, 0) is top-left, (8, 8) is bottom-right
    * show_cursor() takes care of converting position on board to position on screen
* don't let use move cursor off the board

Inputting numbers
================
* main also needs to take '1' to '9' as input
* don't forget about ASCII

Updating the board
==================
* g.board[i][j] contain the number at row i, column j
* various draw_functions redraw the board based on g

Changing the Board
=================
* don't allow users to change nums that came with the board
* when game is started, need to remember which numbers were already placed
* before changing any space, check that space can be changed

Design
======
* factor out as much code as possible (reuse functions)

Move legality
=============
* After changing a number, need to check legality
    * if move is illegal, tell via show_banner()
* Banner does NOT need to persist
    * if I make an illegal move, then a legal move, remove banner
* 3 rules for move to be legal
    * number doesn't already exist in row
    * number doesn't already exist in column
    * number doesn't already exist in 3x3 block

Row and Column
==============
* user just inputted number in g.board[g.y][g.x]
    * user to check g.board[g.y]j[ for 0 < j < 8
    * user to check g.board[i][g.x] for 0 < i < 8
* if number already found, move is illegal
    * check for illegal moves, not wrong moves
    * checking for wrong moves is much harder

3x3 Blocks
==========
* board divided into contiguous 3x3 blocks
* need to check within defined block, not necessarily 3 columns right and 3 rows down from cursor
* given some (y, x), determine coordinates of top-left of block
    * sounds like a job for division and friends
    * then check 3 columns right and 3 columns down
    * only need to check on 3x3 block, not every one

Blanks
======
* user must be able to delete number via KEY_BACKSPACE
* blank represented by 0 in g.board
* can't delete blanks that came with the board
* can't delete numbers that came with the board

Won
===
* game is won if
    * every square filled in
    * every row contains every number
    * every contain contains every number
    * every 3x3 block contains every number
* check if game is won whenever user makes a move
