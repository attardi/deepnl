# -*- coding: utf-8 -*-

"""
NER tagger exploiting a deep neural network.
"""

# standard
import sys

# local
from tagger import Tagger
from reader import TaggerReader
from corpus import *

# ----------------------------------------------------------------------

class ToIOBES(object):
    """Convert from IOB to IOBES notation."""

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        # tokens are lists [form, ..., tag]
        for sent in self.iterable:
            l = len(sent)
            for i, tok in enumerate(sent):
                if  i+1 == l or sent[i+1][-1][0] != 'I':
                    if tok[-1][0] == 'B':
                        tok[-1] = 'S'+tok[-1][1:]
                    elif tok[-1][0] == 'I':
                        tok[-1] = 'E'+tok[-1][1:]
            yield sent

class NerReader(TaggerReader):
    """
    This class reads data from a CoNLL03 corpus and turns it into a format
    readable by the neural network for the NER tagging task.
    """

    def read(self, filename):
        """
        :param filename: the name of a file in CoNLL TSV format.
        """
        return ToIOBES(ConllReader(filename))

# ----------------------------------------------------------------------

class NerTagger(Tagger):
    """Performs NER tagging on sentences."""
    
    def tag(self, sent):
        tags = self.toIOB(self.tag_sequence(sent))
        return zip(sent, tags)

    def toIOB(self, tags):
        """
        Convert back from IOBES to IOB notation.
        """
        res = []
        for tag in tags:
            if tag[0] == 'S':
                res.append('B'+tag[1:])
            elif tag[0] == 'E':
                res.append('I'+tag[1:])
            else:
                res.append(tag)
        return res

