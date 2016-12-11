#!/usr/bin/python
"""
Upgrade to new IOB convention: Inside, Outside, Begin.
"""
# O I -> O B
# I B -> I B
# I I -> I I
# I O -> I O

from __future__ import print_function
import sys
import getopt


def usage():
    print('usage:', sys.argv[0], '[-hr] < inFile ')
    print('  -r   revert to old convention.')
    sys.exit()

try:
    opts, args = getopt.getopt(sys.argv[1:], 'hr')
except getopt.GetoptError:
    usage()

reverse = False

for opt, arg in opts:
    if opt == '-h':
        usage()
    if opt == '-r':
        reverse = True


def main():
    previous = None
    for line in sys.stdin:
        if line == '\n':
            print('\t'.join(previous))
            print()
            previous = None
            continue
        words = line.split()
        word = words[0]
        tag =  words[-1]
        if reverse:
            if tag[0] == 'B' and (previous == None or previous[-1] == 'O'):
                words[-1] = 'I' + tag[1:]
        else:
            if tag[0] == 'I' and (previous == None or previous[-1] == 'O'):
                words[-1] = 'B' + tag[1:]
        if previous:
            print('\t'.join(previous))
        previous = words
    if previous:                # leftover
        print('\t'.join(previous))

main()
