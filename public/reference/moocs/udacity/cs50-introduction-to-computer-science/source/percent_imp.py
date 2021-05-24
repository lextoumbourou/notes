from sys import stdout
from time import sleep

for i in range(1,101):
    stdout.write("\rPercent complete: {0}%".format(i))
    stdout.flush()
    time.sleep(1)

stdout.write("\r \r\n")
