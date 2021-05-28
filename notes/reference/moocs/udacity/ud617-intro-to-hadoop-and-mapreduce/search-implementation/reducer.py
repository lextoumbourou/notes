#!/usr/bin/python
import sys

def reducer():
    old_key = None
    output = None
    for line in sys.stdin:
        word, score, node_id = line.strip().split("\t")
        if old_key != word:
            if output:
                sorted_output = sorted(output, key=lambda x: x[0], reverse=True)
                print "{0}\t{1}".format(old_key, sorted_output)
            old_key = word
            output = []
        output.append(
            (int(score), node_id)
        )
    if output and old_key:
        sorted_output = sorted(output, key=lambda x: x[0], reverse=True)
        print "{0}\t{1}".format(old_key, sorted_output)

if __name__ == '__main__':
    reducer()
