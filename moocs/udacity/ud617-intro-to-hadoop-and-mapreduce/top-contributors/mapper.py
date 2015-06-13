#!/usr/bin/python
import sys
import csv
from datetime import datetime


def mapper():
    """
    For each post, print out the author_id followed by the count (for combining)
    """
    reader = csv.reader(sys.stdin, delimiter='\t', quotechar='"')
    writer = csv.writer(sys.stdout, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)

    for line in reader:
        author_id = line[3]
        print "{0}\t1".format(author_id)

if __name__ == '__main__':
    mapper()
