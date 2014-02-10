#!/usr/bin/python
import sys
import csv
from datetime import datetime
import re

def mapper():
    """
    Print the post length and all the answers to stdout
    """
    reader = csv.reader(sys.stdin, delimiter='\t', quotechar='"')
    writer = csv.writer(sys.stdout, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)

    for line in reader:
        # Ignore any line that's not a forum post (like the header, for example)
        if not line[0].isdigit(): continue
        # Skip lines that don't match the expected format
        if len(line) != 19: continue

        node_id = line[0]
        body = line[4]
        score = line[9]

        split_regex = re.compile('[\s\.\!\?\:\;\"\(\)\<\>\[\]\$\#\=\-\/\,]')
        for word in split_regex.split(body):
            if word:
                print "{0}\t{1}\t{2}".format(word, score, node_id)

if __name__ == '__main__':
    mapper()
