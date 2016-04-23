# -*- coding: utf-8 -*-

"""
POS tagger exploiting a deep neural network.
"""

# standard
import sys
from __future__ import print_function

# local
from network import Network
from tagger import Tagger
from reader import PosReader
from corpus import *

# ----------------------------------------------------------------------

class PosTagger(Tagger):
    """A PosTagger loads the model and performs POS tagging on text."""

    def tag(self, file=sys.stdout):
        """
        :param filename: the file from which to read, stdin if missing.
        """
        reader = PosReader(file)
        writer = ConllWriter()
        for sent in reader:
            print(writer.write(self.tag_sequence(sent)))

