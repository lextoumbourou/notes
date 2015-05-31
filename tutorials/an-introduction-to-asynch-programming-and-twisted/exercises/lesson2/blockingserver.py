"""
Simple blocking server listening on multiple ports to test my async client.

An ugly, simple rewrite of https://github.com/jdavisp3/twisted-intro/blob/master/blocking-server/slowpoetry.py for
memory retention.
"""

import socket
import sys
import time


if len(sys.argv) != 3:
    print "Usage: blockingserver.py <port> <filename>"
    exit(1)

HOST = '127.0.0.1'
PORT = int(sys.argv[1])
FILENAME = sys.argv[2]

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()

print 'Listening on {}:{}'.format(HOST, PORT)

f = open(FILENAME)

while True:
    text = f.read(1024)
    if not text:
        break
    conn.sendall(text)
    time.sleep(3)

conn.close()
