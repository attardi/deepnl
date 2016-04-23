#!/usr/bin/python

"""
Tokenize a Twitter corpus in CoNLL 2013 format.

Usage:
   tweet-tokenize.py [options] < CoNLL20113-file

Optons:

   -h		print this help message
   -l language	select corpus language (default english)

"""

import os
import sys
import getopt
from __future__ import print_function

# Tanl directory
tanl = '/project/piqasso/QA/Tanl/'

# Where to find data
data = tanl + 'data/'

# import Tanl modules

sys.path.append(tanl + 'bin/')

from SentenceSplitter import *
from Tokenizer_it import *

### CLI INTERFACE ############################################################

def show_help():
    print(__doc__, end='')

def show_usage(scriptname):
    print('Usage: %s [options] [file]' % scriptname, file=sys.stderr)

def show_suggestion(scriptname):
    print('Try \'%s --help\' for more information.' % scriptname, file=sys.stderr)

def main():
    scriptname = os.path.basename(sys.argv[0])

    try:
        long_opts = ['language=', 'help', 'usage']
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'l:h', long_opts)
    except getopt.GetoptError:
        show_usage(scriptname)
        show_suggestion(scriptname)
        sys.exit(1)
    
    lang = 'english'

    for opt, arg in opts:
        if opt in ('-l', '--language'):
            lang = arg
        elif opt in ('-h', '--help'):
            show_help()
            return
        elif opt == '--usage':
            show_usage(scriptname)
            return

    # Tanl modules
    splitterModel = data + 'split/sentence/' + lang + '.punkt'
    if os.path.exists(splitterModel):
        print("No such model:" + splitterModel)
        return
    t0 = SentenceSplitter(splitterModel)
    t1 = Tokenizer()

    # Field containig tweets
    text_field = 3

    for line in sys.stdin:
        fields = line.split('\t')
        p0 = t0.pipe([fields[text_field]]) # SentenceSplitter
        p1 = t1.pipe(p0)                   # Tokenizer
        tokens = []
        for t in p1:
            form = t['FORM']
            if form != '\n':
                tokens.append(form)
        fields[text_field] = ' '.join(tokens)
        print('\t'.join(fields))

if __name__ == '__main__':
    main()
