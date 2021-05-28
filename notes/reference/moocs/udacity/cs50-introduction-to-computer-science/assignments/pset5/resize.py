#!/usr/bin/python
"""
resize.py

Computer Science 50 in Python
Problem Set 5

Resizes an image by a factor of n
Usage: resize.py n infile outfile
"""
import Image
from sys import argv, exit
import array

if __name__ == "__main__":
    if len(argv) < 4:
        print "Usage: resize.py n infile outfile"
        exit(1)

    try:
        n = abs(int(argv[1]))
    except ValueError:
        print "n must be an integer"
        exit(2)

    infile = argv[2]
    outfile = argv[3]

    try:
        in_img = Image.open(infile)
    except IOError, e:
        print "Cannot open file: ", e

    width, height = in_img.size

    # Get new width and height
    new_width, new_height = width * n, height * n

    # Resize image
    out_img = in_img.resize((new_width, new_height), Image.NEAREST) 

    # Save the image
    try:
        out_img.save(outfile)
    except IOError:
        print "Could not save file."
        exit(3)
