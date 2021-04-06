"""
Simple async client.

A simplified rewrite of https://github.com/jdavisp3/twisted-intro/blob/master/async-client/get-poetry.py from memory.
"""
import socket
import select
import sys
import errno

if len(sys.argv) != 3:
    print "Usage: asyncclient.py <port1> <port2>"
    exit(1)

HOST = '127.0.0.1'

sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock1.connect((HOST, int(sys.argv[1])))
sock1.setblocking(0)

sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock2.connect((HOST, int(sys.argv[2])))
sock2.setblocking(0)

sockets = [sock1, sock2]

while sockets:
    data = ''

    rlist, _, _ = select.select(sockets, [], [])

    for sock in rlist:
        try:
            data = sock.recv(1024)
        except socket.error as e:
            if e == errno.EWOULDBLOCK:
                break
            raise
        else:
            if not data:
                sockets.remove(sock)
                sock.close()
                break

            print "**** Start received data *****"
            print data
            print "**** End received data *****"
