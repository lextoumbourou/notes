"""
whodunit.py

Computer Science 50 in Python
Problem Set 5

Decypher message hidden in BMP.
Run without arguments for usage.
"""
from sys import argv, exit
import struct
import Image

if __name__ == "__main__":
    # Ensure proper usage
    if not len(argv) == 3:
        print "Usage: copy infile outfile"
        exit(1)

    # Remember filenames
    infile = argv[1]
    outfile = argv[2]

    # Open input file
    try:
        img = Image.open(infile)
    except IOError:
        print "Could not open {0}".format(infile)
        exit(2)

    # Get all the pixels
    pixels = list(img.getdata())

    new_data = []
    for pixel in pixels:
        # Change red to white
        if pixel == (255, 0, 0):
            new_data.append((0, 0, 0))
        # Change white to black
        elif pixel == (255, 255, 255):
            new_data.append((0, 0, 0))
        # The rest should be the message
        else:
            new_data.append((255, 0, 0))

    # Create a new file and attempt to save to disk
    new_img = Image.new(img.mode, img.size)
    new_img.putdata(new_data)
    try:
        new_img.save(outfile)
    except IOError:
        print "Failed to write to disk"
