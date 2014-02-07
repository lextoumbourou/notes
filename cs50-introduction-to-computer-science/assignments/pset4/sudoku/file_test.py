from struct import unpack

def get_board_from_bin(level, board_num):
    """
    Load a board of 81 ints from the appropriate board
    and return an array
    """
    filename = "{0}.bin".format(level)
    try:
        f = open(filename, "rb")
    except IOError:
        return False

    # We seek through the file finding the 81 numbers
    # at each position
    seek_min = (board_num - 1) * 81
    seek_max = seek_min + 81

    # Iterate through the n to n+81 range
    # populating our 9*9 multidimensional array
    key = 0
    board = [[] for i in range(0, 9)]
    for i in range(seek_min, seek_max):
        f.seek(i * 4)
        # Collect the integer at this point in the file
        board_int = unpack("@i", f.read(4))[0]
        board[key].append(board_int)
        # Once we've gone through 9 values, we're going to 
        # increase the key to continue populating the array
        if (i + 1) % 9 == 0:
            key += 1

    f.close()

    return board

print get_board_from_bin('l33t', 4)
