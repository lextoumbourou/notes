#!/usr/bin/python
"""
cracker.py

Computer Science 50 in Python (Hacker Edition)
Problem Set 2

"""

import crypt
import sys

class Cracker:
    def __init__(self, hash):
        # Get first sys argument which should be the hash to crack
        self.hash = hash
        # First two chars in hash are the Salt
        self.salt = self.hash[:2]
        # Default charset, can be extended if need be
        self.charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#"
        # Max password size
        self.max_size = 10

    def compare_hash(self, pwd):
        """Compare inputted hash to classes hash, exit if match"""
        test_hash = crypt.crypt(pwd, self.salt)
        if test_hash == self.hash:
            print "{0}".format(pwd)
            sys.exit()
    
    def dictionary(self):
        """
        Iterate through each line in the local list of words and compare the password
        against them
        """
        try:
            dict_hand = open("/usr/share/dict/words", "r")
        except IOError:
            print "Dictionary not found."
            return False

        # Go through each word in the dictionary, as compare the hashed password outputted to the original
        for line in dict_hand:
            self.compare_hash(line.strip())

    def brute_force(self):
        for width in range(1, self.max_size+1):
            self._pw_builder(0, width, base_str="")

        # If the program hasn't exited, then no match has been found
        print "Failed to brute force password."
		
    def _pw_builder(self, position, width, base_str):
        """
        Iterate through each character in the charset, 
        calling itself to add a character to the base string once complete
        """
        for char in self.charset:
            if (position < width-1):
                # If we are on the next position, we need to call method again, adding an extra
                # character to iterate over
                self._pw_builder(position + 1, width, base_str + char)

            self.compare_hash(base_str + char)

def get_args():
    if len(sys.argv) is not 2:
        sys.exit(1)
    else:
        return sys.argv[1]

if __name__ == '__main__':
    crack = Cracker(get_args())
    crack.dictionary()
    crack.brute_force()
