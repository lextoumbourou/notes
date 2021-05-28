#!/usr/bin/python
"""
sudoku.py

Computer Science 50 in Python
Problem Set 4

Implements the game of Sudoku
"""
from os import system
from struct import unpack
from sys import argv, exit
import random
import curses
import time

TITLE = "Sudoku"
AUTHOR = "Lex Toumbourou"

class Board():
    def __init__(self, board_num, level):
        self.board_num = board_num
        self.level = level

        
    def load(self):
        """
        Load a board of 81 ints from the appropriate board
        and return an array
        """
        filename = "{0}.bin".format(self.level)
        try:
            f = open(filename, "rb")
        except IOError, e:
            print e
            return False

        # We seek through the file finding the 81 numbers
        # at each position
        seek_min = (self.board_num - 1) * 81
        seek_max = seek_min + 81

        # Iterate through the n to n+81 range
        # populating our 9*9 multidimensional array
        key = 0
        self.built_in_values = []
        board = [[] for i in range(0, 9)]
        x_count = 0
        for i in range(seek_min, seek_max):
            f.seek(i * 4)
            # Collect the integer at this point in the file
            board_int = unpack("@i", f.read(4))[0]
            board[key].append(board_int)
            # If the int is more than 0, we have position that can't be changed
            if board_int > 0:
                self.built_in_values.append( (key, x_count ) )
            x_count += 1
            # Once we've gone through 9 values, we're going to 
            # increase the key to continue populating the array
            if (i + 1) % 9 == 0:
                key += 1
                x_count = 0 

        f.close()
        self.board = board

        return True

    def startup(self):
        """Start up ncurses"""
        self.screen = curses.initscr()

        curses.start_color()
        curses.noecho()
        curses.curs_set(2)
        self.screen.keypad(1)

        # Initialise color pairs
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_RED)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)

        self.maxy, self.maxx = self.screen.getmaxyx()
        self.maxy = self.maxy - 2
        
        return True

    def redraw_all(self):
        """(Re)draw everything on the screen"""
        # Reset curses
        curses.endwin()
        self.screen.refresh()

        # Clear the screen
        #self.clear()

        # Re-draw everything
        self.draw_borders()
        self.draw_grid()
        self.draw_logo()
        self.draw_numbers()

        self.show_cursor()

    def draw_borders(self):
        #self.screen.addstr(0, 0, "y is {0} and x is {1}".format(self.y, self.x))

        # Draw borders
        for i in range(0, self.maxx):
            self.screen.addstr(0, i, " ", curses.color_pair(1))
            self.screen.addstr(self.maxy, i, " ", curses.color_pair(1))

        ## Draw header
        header = "{0} by {1}".format(TITLE, AUTHOR) 
        self.screen.addstr(0, (self.maxx - len(header)) / 2, header, curses.color_pair(1))
        
        ## Draw footer
        self.screen.addstr(self.maxy, 1, "[N]ew Game   [R]estart Game", curses.color_pair(1))
        self.screen.addstr(self.maxy, self.maxx - 13, "[Q]uit Game", curses.color_pair(1))
        

    def draw_grid(self):
        # Determine where top-left corner of board belongs
        self.top = self.maxy / 2 - 7
        self.left = self.maxx / 2 - 30

        # Print grid
        for i in range(0, 3):
            self.screen.addstr(self.top + 0 + 4 * i, self.left, "+-------+-------+-------+")
            for n in range(1, 4):
                self.screen.addstr(self.top + n + 4 * i, self.left, "|       |       |       |")
        self.screen.addstr(self.top + 4 * 3 , self.left, "+-------+-------+-------+")

        # Remind user of level and #
        reminder = "   playing {0} #{1}".format(self.level, self.board_num)
        self.screen.addstr(self.top + 14, self.left + 25 - len(reminder), reminder)
    
    def draw_logo(self):
        """
        Draw game's logo. Must be called after draw_grid has been called
        at least once.
        """
        logo_pos_top = self.top + 2
        logo_pos_left = self.left + 30

        # draw logo
        self.screen.addstr(logo_pos_top + 0, logo_pos_left, "               _       _          ");
        self.screen.addstr(logo_pos_top + 1, logo_pos_left, "              | |     | |         ");
        self.screen.addstr(logo_pos_top + 2, logo_pos_left, " ___ _   _  __| | ___ | | ___   _ ");
        self.screen.addstr(logo_pos_top + 3, logo_pos_left, "/ __| | | |/ _` |/ _ \\| |/ / | | |");
        self.screen.addstr(logo_pos_top + 4, logo_pos_left, "\\__ \\ |_| | (_| | (_) |   <| |_| |");
        self.screen.addstr(logo_pos_top + 5, logo_pos_left, "|___/\\__,_|\\__,_|\\___/|_|\\_\\\\__,_|");

        # sign logo
        signature = "by {0}".format(AUTHOR)
        self.screen.addstr(logo_pos_top + 7, logo_pos_left + 35 - len(signature) - 1, signature)

    def draw_numbers(self, built_in_color=1, user_color=4):
        """
        Draw game's numbers. Must be called after draw_grid has been
        called at least once.
        """
        for i in range(0, 9):
            for j in range(0, 9):
                y = self.top + i + 1 + i/3
                x = self.left + 2 + 2*(j + j/3)
                char = 0
                if self.board[i][j]:
                    char = str(self.board[i][j])
                else:
                    char = "."

                if (i,j) in self.built_in_values:
                    self.screen.addstr(y, x, char, curses.color_pair(built_in_color))
                else:
                    self.screen.addstr(y, x, char, curses.color_pair(user_color))

    def handle_signal(self, signum):
        """Handle signals (ie, a resizing)"""
        # Handle a change in the window (i.e., a resizing)
        if (signum == SIGWINCH):
            self.redraw_all()

    def show_cursor(self):
        try:
            self.screen.move(self.curs_y, self.curs_x)
        except AttributeError:
            self.curs_y = self.top + 4 + 1 + 4/2
            self.curs_x = self.left + 2 + 2*(4 + 4/3)
            self.show_cursor()

    def _rel_y(self):
        output = self.curs_y - self.top
        if output >= 5 and output <= 7:
            output -= 1
        if output >= 9 and output <= 11:
            output -= 2
        return output

    def _rel_x(self):
        output = (self.curs_x - self.left) / 2
        if output >= 5 and output <= 7:
            output -= 1
        if output >= 9 and output <= 11:
            output -= 2
        return output


    def move_cursor(self, move):
        """
        Move and show cursor at x, y
        """
        # Check if edge of board
        if move == curses.KEY_LEFT:
            if self._rel_x() > 1:
                # Check divider
                if self._rel_x() == 4 or \
                   self._rel_x() == 7:
                    self.curs_x -= 4
                else:
                    self.curs_x -= 2
            else:
                self.curs_x += 9 * 2 + 2 
        elif move == curses.KEY_RIGHT:
            if self._rel_x() < 9:
                if self._rel_x() == 3 or \
                   self._rel_x() == 6:
                    self.curs_x += 4
                else:
                    self.curs_x += 2
            else:
                self.curs_x -= 9 * 2 + 2 
        elif move == curses.KEY_UP:
            if self._rel_y() > 1:
                # Check divider 
                if self._rel_y() == 4 or \
                   self._rel_y() == 7:
                    self.curs_y -= 2
                else:
                    self.curs_y -= 1
            else:
                self.curs_y += 9 + 1
        elif move == curses.KEY_DOWN:
            if self._rel_y() < 9:
                # Check divider 
                if self._rel_y() == 3 or \
                   self._rel_y() == 6:
                    self.curs_y += 2
                else:
                    self.curs_y += 1
            else:
                self.curs_y -= 9 + 1


        self.screen.move(self.curs_y, self.curs_x)

    def delete_pos(self):
        y = self._rel_y() - 1
        x = self._rel_x() - 1

        # User can only change values that aren't already built-in
        if (y, x) in self.built_in_values:
            self.screen.addstr(20, 20, "You can't delete built-in values.")
            return False

        self.board[y][x] = 0

    def update_pos(self, choice):
        y = self._rel_y() - 1
        x = self._rel_x() - 1

        # User can only change values that aren't already built-in
        if (y, x) in self.built_in_values:
            return False

        # Check that number isn't in row
        if choice in self.board[y]:
            self.screen.addstr(20, 20, "{0} is already in the row.   ".format(choice))
            return False

        # Check that the number isn't in column
        for i in range(0, 9):
            if self.board[i][x] == choice:
                self.screen.addstr(20, 20, "{0} is already in the column ".format(choice))
                return False

        # Get closest top boundary
        for i in range(y, y + 3):
            if (i + 1) % 3 == 0:
                y_corner = i
                break

        # Get closest right boundary
        for i in range(x, x + 3):
            if (i + 1) % 3 == 0:
                x_corner = i
                break

        # Go backwards, bottom to top, right to left, looking for the current choice in the boundary
        for i in range(y_corner - 2, y_corner + 1):
            for j in range(x_corner - 2, x_corner + 1):
                if self.board[i][j] == choice:
                    self.screen.addstr(20, 20, "{0} is already in the boundary ".format(choice))
                    return False
        
        # To do: work out a better way to clear the screen 
        self.screen.addstr(20, 20, "                       ".format(choice))
        self.board[y][x] = choice

    def has_won(self):
        """If all spaces are filled, then the game is over"""
        for i in range(0, 9):
            for j in range(0, 9):
                if self.board[i][j] == 0:
                    return False

        return True

    def finish(self):
        """Display congratulatory banner, and turn all the squares green"""

        self.screen.refresh()
        self.draw_grid()
        self.draw_numbers(3, 3)
        # Hide the cursor
        curses.curs_set(0)

    def restart_game(self):
        """(Re)start the current game, returning True if successful"""
        # Reload board
        if not self.load():
            return False

        # Redraw board
        self.draw_grid()
        self.draw_numbers()

        # Move cursors to board's center
        #self.y = self.x = 4
        #self.show_cursor()

        # Remove the log, if any
        #os.remove("log.txt")

        return True

    def shutdown(self):
        """Shut down interface"""
        curses.endwin()


def hide_banner(self):
    """Hide banner"""
    for i in range(0, self.max_x):
        screen.addstr(self.top + 16, i, " ")

def log_move(self):
    """
    Log input and board's state to log.txt to
    faciliate automated tests
    """
    try: 
        f = open("log.txt", "rw")
    except IOError:
        return False

    # Log input
    f.write("{0}\n".format(char))

    # Log board
    for i in range(0, 9):
        for j in range(0, 9):
            f.write("{0}".format(self.board[i][j]))
        f.write("\n")

    f.close()

def show_banner(self, b):
    """
    Show a banner. Must be called after show_grid has
    been called at least once
    """
    # Determine where top-left corner of board belongs
    self.addstr(self.top + 16, self.left + 64 - strlen(b), b)

if __name__ == '__main__':
    #curses.noecho()
    #curses.curs_set(0)
    #screen.keypad(1)

    usage = "Usage: sudoku n00b|l33t [#]"

    # Ensure number of arguments is as expected
    if len(argv) < 2:
        print(usage)
        exit(1)

    # ensure that level is valid
    if argv[1] == "debug" or \
       argv[1] == "n00b"  or \
       argv[1] == "l33t":
        level = argv[1] 
    else:
        print(usage)
        exit(2)

    # n00b and l33t levels have 1024 boards, debug level has 9
    max = 1024 if level != "debug" else 9

    if len(argv) is 3:
        try:
            board_num = int(argv[2])
        except ValueError:
            print("Board number isn't a number.")
            exit(3)

        if board_num < 1 or board_num > max:
            print("That board # does not exist!")
            exit(4)
    else:
        # seed PRNG with $ so that we get same sequence of boards
        board_num = random.randint(1, 1024)
    
    # load the board from provided binary files
    board = Board(board_num, level)

    if not board.load():
        print "Failed to open board"
        exit(6)

    if not board.startup():
        print "Failed to start ncurses"
        exit(5)

    board.redraw_all()

    while True:
        # refresh the screen
        board.screen.refresh()

        # get the user's input
        choice = board.screen.getch()

        # See if movement
        if choice == curses.KEY_UP   or \
           choice == curses.KEY_DOWN or \
           choice == curses.KEY_LEFT or \
           choice == curses.KEY_RIGHT:
            board.move_cursor(move=choice)
            continue

        # Check if number
        try:
            choice = int(chr(choice))
            if choice >= 1 and choice <= 9:
                board.update_pos(choice)
                board.redraw_all()
                if board.has_won():
                    board.finish()
                continue
        except ValueError:
            pass

        try:
            choice = chr(choice).lower()
        except AttributeError:
            pass

        # process input
        # Make a new random number for creating the board
        if choice == 'x':
            board.delete_pos()
            board.redraw_all()
            continue
        elif choice == 'n':
            board.board_num = random.randint(1, 1024)
            if not board.restart_game():
                board.shutdown()
                print("Could not load board from disk")
                exit(6)
        # Since they're restarting
        # the random number shouldn't be regenerated
        elif choice == 'r':
            if not board.restart_game():
                board.shutdown()
                print("Could not load board from disk")
                exit()
        elif choice == 'q':
            board.shutdown()
            exit()
        else:
            continue
