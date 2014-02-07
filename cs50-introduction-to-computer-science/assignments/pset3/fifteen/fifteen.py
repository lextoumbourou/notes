"""
fifteen.py

Computer Science 50 in Python
Problem Set 3
"""


from sys import argv, stdout
from time import sleep

DIM_MIN = 3
DIM_MAX = 9

def clear():
    """Clear screen using ANSI escape sequences."""
    stdout.write("\033[2J")
    stdout.write("\033[{0};{1}H".format(0, 0))

def greet():
    """Greet player"""
    clear()
    print("WELCOME TO THE GAME OF FIFTEEN")
    sleep(1)

class Game:
    def __init__(self, dims):
        """
        Initialise the game's board with tiles numbered 1 through d*d - 1
        (i.e., fills 2D array with values but does not actually print them).
        """
        self.board = []
        self.blank = "_"
        total = (dims * dims) - 1
        total_left = total

        for i in range(dims):
            self.board.append([])
            for j in range(dims):
                if total_left is 0:
                    self.board[i].append(self.blank)
                    continue

                if total % 2 is 0:
                    self.board[i].append(str(total_left))
                else:
                    if total_left > 2:
                        self.board[i].append(str(total_left))
                    # To do: fix this up to check if last row and odd
                    elif total_left is 2:
                        self.board[i].append("1")
                    elif total_left is 1:
                        self.board[i].append("2")
                    
                total_left -= 1

        print self.board

    def draw(self):
        """Print the board in its current state"""
        output = ""
        for c, i in enumerate(self.board):
            for j in self.board[c]:
                output += "{0:>2} ".format(j)
            output += "\n"

        print output

    def _swap(self, tile):
        self.board[self.c_row][self.c_col] = self.blank
        self.board[self.b_row][self.b_col] = tile

        return True

    def move(self, tile):
        """
        If title borders empty space, moves tile and return true, else
        """
        self.c_col = None
        self.c_row = None
        
        # Get tile position
        for row, vals in enumerate(self.board):
            for col, val in enumerate(vals):
                if val == str(tile):
                    self.c_col = col
                    self.c_row = row
                if val == self.blank:
                    self.b_col = col
                    self.b_row = row

        # Try to swap values
        if (self.c_col or self.c_row) is None:
            return False

        if self.b_row is self.c_row:
            # If same row, ensure col vals are only 1 apart 
            if abs(self.c_col - self.b_col) is  1:
                return self._swap(tile)

        # If the rows are further than 1 place apart
        if abs(self.b_row - self.c_row) is 1:
            # No diagonal
            if self.b_col is self.c_col:
                return self._swap(tile)

        return False
       
    def won(self):
        """
        Return true if game is won (i.e., board is in winning configuration),
        else false.
        """
        for row, vals in enumerate(self.board):
            for col, v in enumerate(vals):
                try:
                    if v > self.board[row][col+1]:
                        return False
                except ValueError:
                    pass

        return True

if __name__ == '__main__':
    # greet user with instructions
    greet()

    # Ensure proper usage
    if (len(argv) is not 2):
        print("Usage: fifteen d\n")
        exit(1)

    try:
        dims = int(argv[1])
    except ValueError:
        print "Argument not a number."
        exit(3)

    if dims < DIM_MIN or dims > DIM_MAX:
        print("Board must be between {0} and {1}, inclusive.".format(DIM_MIN, DIM_MAX))
        exit(2)

    # Initialise the board
    game = Game(dims)

    # Accept moves until game is won
    while True:
        # clear the screen
        clear()

        # draw the current state of the board
        game.draw()

        # check for win
        if game.won():
            print("ftw!")
            break

        # prompt for move
        tile = raw_input("Tile to move: ")

        # move if possible, else report illegality
        if not game.move(tile):
            print("Illegal move.")
            sleep(1)

        # sleep thread for animation's sake

    exit(0)
