#!/usr/bin/python
import sys


def reducer():
    """
    Output the top 10 contributers
    """
    output = {}
    count = 0
    for line in sys.stdin:
        author, count = line.strip().split("\t")
        output[author] = output.get(author, 0) + int(count)

    results = sorted(output.iteritems(), key=lambda x: x[1], reverse=True)
    for author, count in results[:10]:
        print "{0}\t{1}".format(author, count)

if __name__ == '__main__':
    reducer()
