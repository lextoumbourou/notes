from sys import stdout
from time import sleep

def draw_board(x=None, y=None):
    # print the first new line
    output = " "
    # print the top row
    for n in range(1,11):
        output += "{0} ".format(n)

    output += "\n"

    # print row of holes, with letters in leftmost columns
    for l in "ABCDEFG":
        output += l + " "
        for n in range(1,11):
            if n == x and l == y:
                output += "x"
            else:
                output += "o"
            output += " "
        output += "\n"
    output += "\r"

    return output

stdout.write(draw_board(x=4, y='B'))





