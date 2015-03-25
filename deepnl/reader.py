#!/usr/env python
# -*- coding: utf-8 -*-
#cython: embedsignature=True

"""
Classes for reading various types of corpora.
"""

# standard
import os
import logging
import numpy as np
from collections import Counter
import gzip

# local
from corpus import *
from embeddings import Plain

class Reader(object):
    """
    Abstract class for corpus readers.
    """
    
    # force class to be abstract
    #__metaclass__ = abc.ABCMeta

    def create_vocabulary(self, sentences, size, min_occurrences=3):
        """
        Create vocabulary from sentences.
        :param sentences: an iterable on sentences.
        :param size: size of the vocabulary
        :param min_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        Sentence tokens are lists [form, ..., tag]
        """
        c = Counter()
        for sent in sentences:
            for token, in sent:
                c[token] += 1
        common = c.most_common(size)
        words = [w for w, n in common if n >= min_occurrences]
        return words

    def load_vocabulary(self, filename):
        return Plain.read_vocabulary(filename)

# ----------------------------------------------------------------------

class TextReader(Reader):
    """
    Reads sentences from tokenized text file.
    """
    
    def __init__(self, variant=None):
        """
        :param sentences: A list of lists of tokens.
        """
        super(TextReader, self).__init__()
        self.variant = variant

    def read(self, filename):
        """
        :param filename: name of the file from where sentences are read.
            The file should have one sentence per line, with tokens
            separated by white spaces.
        :return: an iterable over sentences, which can be iterated over several
            times.
        """
        class iterable(object):
            def __iter__(self):
                if filename.endswith('.gz'):
                    file = gzip.GzipFile(filename, 'rb')
                else:
                    file = open(filename, 'rb')
                for line in file:
                    sent =  unicode(line, 'utf-8').split()
                    if sent:
                        yield sent
                file.close()

        return iterable()

    def sent_count(self):
        return len(self.sentences)

# ----------------------------------------------------------------------

class TaggerReader(Reader):
    """
    Abstract class extending TextReader with useful functions
    for tagging tasks. 
    """
    
    # force class to be abstract
    #__metaclass__ = abc.ABCMeta

    def __init__(self):
        super(TaggerReader, self).__init__()
        # self.sentence_count = len(sentences) if sentences else 0

    def read(self, filename):
        return ConllReader(filename)

    # def sent_count(self):
    #     return self.sentence_count

    def create_vocabulary(self, sentences, size, min_occurrences=3):
        """
        Create vocabulary and tag set from sentences.
        :param sentences: an iterable on sentences.
        :param size: size of the vocabulary
        :param min_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        Sentence tokens are lists [form, ..., tag]
        """
        c = Counter()
        tags = set()
        for sent in sentences:
            for token, in sent:
                c[token[0]] += 1
                tags.add(token[-1])  # assumes that tag is last token field
        common = c.most_common(size)
        words = [w for w, n in common if n >= min_occurrences]
        return words, tags

    def create_tagset(self, sentences):
        """
        Create tag set from sentences.
        :param sentences: an iterable over sentences.
        """
        tags = set()
        for sent in sentences:
            for token in sent:
                tags.add(token[-1])  # assumes that tag is last token field
        return tags
    
# ----------------------------------------------------------------------

class PosReader(TaggerReader):
    """
    This class reads data from a POS corpus and turns it into a representation
    for use by the neural network for the POS tagging task.
    """
    
    def __init__(self):
        self.rare_tag = None
        super(PosReader, self).__init__()

# ----------------------------------------------------------------------

class TweetReader(Reader):
    """
    Reader for tweets in SemEval 2013 format, one tweet per line consisting  of:
    SID	UID	polarity	tokenized text
    264183816548130816      15140428        positive      Gas by my house hit $3.39!!!! I'm going to Chapel Hill on Sat. :)
    """
    polarity_field = 2
    text_field = 3

    def __init__(self, ngrams=1, variant=None):
        """
	:param ngrams: the lenght of ngrams to consider
        :param filename: the name of the file containing tweets. The file should have one tweet per line.
	:param variant: whether to use native, or SENNA or Polyglot conventions
        """
        super(TweetReader, self).__init__()
	self.ngrams = ngrams
        self.variant = variant
        self.sentences = []
        self.polarities = []

    def read(self, filename):
        with open(filename, 'rb') as f:
            for line in f:
                tweet = unicode(line, 'utf-8').split('\t')
                if tweet[TweetReader.polarity_field] == 'positive':
                    polarity = 1
                if tweet[TweetReader.polarity_field] == 'negative':
                    polarity = -1
                else:
                    continue    # CHECKME: skip tweets with no polarity
                self.sentences.append(tweet[TweetReader.text_field].split())
                self.polarities.append(polarity)
                    
    def acceptable(self, token):
        """Simple criteron to accept a token as part of a phrase, rejecting
        punctuations or common short words.
        """
        return len(token) > 2

    def create_vocabulary(self, sentences, size=None, min_occurrences=3):
        """
        Generates a list of all ngrams from the given sentences.
        
        :param sentences: an iterable on sentences.
        :param size: Max number of tokens to be included in the dictionary.
        :param min_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        """
        
        # unigrams
        ngrams = [[token for sent in self.sentences for token in sent]]
        # multigrams
	for n in xrange(2, self.ngrams + 1):
            ngrams.append([])
	    for sent in self.sentences:
	    	for i in xrange(len(sent) + 1 - n):
                    phrase = sent[i:i+n]
                    accept = True
                    for tok in phrase:
                        if not acceptable(tok):
                            accept = False
                            break
                    if accept:
                        ngrams[-1].append(' '.join(phrase))
        return ngrams
