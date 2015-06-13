import math
import random
import subprocess
import time

FUZZ_FACTOR = 250
FILE_TO_FUZZ = 'burn.mp3'
FUZZ_OUTPUT = 'fuzzed-burn.mp3'
PLAYER = 'afplay'
NUM_TESTS = 10000


for i in range(NUM_TESTS):
    buf = bytearray(open(FILE_TO_FUZZ, 'rb').read())

    numwrites = random.randrange(
        math.ceil((float(len(buf)) / FUZZ_FACTOR))) + 1

    for j in range(numwrites):
        rbyte = random.randrange(256)
        rn = random.randrange(len(buf))
        buf[rn] = '%c' % (rbyte)

    open(FUZZ_OUTPUT, 'wb').write(buf)

    process = subprocess.Popen([PLAYER, FUZZ_OUTPUT])

    time.sleep(1)
    crashed = process.poll()
    if not crashed:
        process.terminate()
