# -*- coding: utf-8 -*-

"""
NER tagger exploiting a deep neural network.
"""

# standard
import sys
from itertools import izip

# local
from tagger import Tagger
from reader import TaggerReader
from corpus import *

# ----------------------------------------------------------------------

class ToIOBES(object):
    """Convert from IOB to IOBES notation:
    Begin
    Inside
    Outside
    Single
    End
    """

    def __init__(self, iterable, tagField):
        self.iterable = iterable
        self.tagField = tagField

    def __iter__(self):
        # tokens are lists [form, ..., tag]
        for sent in self.iterable:
            l = len(sent)
            for i, tok in enumerate(sent):
                if  i+1 == l or sent[i+1][self.tagField][0] != 'I':
                    if tok[self.tagField][0] == 'B':
                        tok[self.tagField] = 'S'+tok[self.tagField][1:]
                    elif tok[self.tagField][0] == 'I':
                        tok[self.tagField] = 'E'+tok[self.tagField][1:]
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
        return ToIOBES(ConllReader(filename), self.tagField)

# ----------------------------------------------------------------------

class NerTagger(Tagger):
    """Performs NER tagging on sentences."""
    
    def tag(self, sent, tagField=-1):
        tags = self.toIOB(super(NerTagger, self).tag(sent))
        for tok,tag in izip(sent, tags):
            tok[tagField] = tag
        return sent

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

