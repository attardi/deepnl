#!/usr/env python
# -*- coding: utf-8 -*-
#cython: embedsignature=True

"""
Classes for reading/writing various types of corpora.
"""

# standard
from __future__ import print_function
import sys
import codecs

class ConllReader(object):
    """
    An iterator over sentences read from a file in CoNLL TSV format.
    If the input is from a file, it can be iterated several times.
    """
    def __init__(self, filename=None):
        self.filename = filename

    def __iter__(self):
        if self.filename:
            file = codecs.open(self.filename, 'r', 'utf-8', errors='ignore')
        else:
            file = codecs.getreader('utf-8')(sys.stdin)
        sent = []
        for line in file:
            line = line.strip()
            if line:
                sent.append(line.split('\t'))
            else:
                yield sent
                sent = []
        if sent:                # just in case
            yield sent
        if self.filename:
            file.close()

    def count(self):
        """
        :return: the number of sentences.
        """
        empty_lines = 0
        buf_size = 1024 * 1024
        file = open(self.filename)
        read_f = file.read # loop optimization
        while True:
            buf = read_f(buf_size)
            if not buf:
                break
            # FIXME: this fails if \n\n is split between two buffers
            empty_lines += buf.count('\n\n') # empty lines
        if self.filename:
            file.close()
        return empty_lines

# ----------------------------------------------------------------------

class ConllWriter(object):
    """
    Prints one token per line as token<tab>tag,
    with sentence separated by empty line.
    """

    @classmethod
    def write(cls, sent):
        """
        Prints a sentence to stdout in TSV format
        
        :param sent: the sentence to write.
        """
        for token in sent:
            print('\t'.join([item.encode('utf-8') for item in token]))
        print()

# ----------------------------------------------------------------------

class SrlWriter(object):

    @classmethod
    def write(cls, sent):
        """
        :param sent: must be of type SRLAnnotatedSentence
        """
        print(' '.join(sent.tokens).encode('utf-8'))
        for predicate, arg_structure in sent.arg_structures:
            print(predicate.encode('utf-8'))
            for label in arg_structure:
                argument = ' '.join(arg_structure[label])
                line = '\t%s: %s' % (label, argument)
                print(line.encode('utf-8'))
        print()

# ----------------------------------------------------------------------

class TsvReader(object):
    """
    An iterator over examples read from a file in TSV format.
    If the input is from a file, it can be iterated several times.
    """
    def __init__(self, filename=None):
        self.filename = filename

    def __iter__(self):
        if self.filename:
            file = codecs.open(self.filename, 'r', 'utf-8', errors='ignore')
        else:
            file = codecs.getreader('utf-8')(sys.stdin)
        for line in file:
            line = line.strip()
            if line:
                yield line.split('\t')
        if self.filename:
            file.close()
