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
        """
        :return: an iterator on sentences.
        """
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
        common = c.most_common(size) # common is a list of pairs
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
	:param ngrams: the length of ngrams to consider
	:param variant: whether to use native, or SENNA or Polyglot conventions
        """
        super(TweetReader, self).__init__()
	self.ngrams = ngrams
        self.variant = variant
        self.sentences = []
        self.polarities = []

    def read(self, filename):
        for tweet in TsvReader(filename):
            if tweet[TweetReader.polarity_field] == 'positive':
                polarity = 1
            elif tweet[TweetReader.polarity_field] == 'negative':
                polarity = -1
            else:
                polarity = 0
            self.sentences.append(tweet[TweetReader.text_field].split())
            self.polarities.append(polarity)
        return self.sentences
                    
    def acceptable(self, token):
        """Simple criteron to accept a token as part of a phrase, rejecting
        punctuations or common short words.
        """
        return len(token) > 2

    # discount to avoid phrases with very infrequent words
    delta = 1

    def create_vocabulary(self, tweets, size=None, min_occurrences=3, threshold=0.1):
        """
        Generates a list of all ngrams from the given tweets.
        
        :param tweets: an iterable on tweets.
        :param size: Max number of tokens to be included in the dictionary.
        :param min_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        :param threshold: minimum bigram score
        :return: list of ngrams (joined by '_'), list of bigrams
        and list of trigrams
        """
        
        # Use PMI-like score for selecting collocations:
        # score(x, y) = (count(x,y) - delta) / count(x)count(y)
        # @see Mikolov et al. 2013. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013
        # unigrams
        unigramCount = Counter(token for tweet in tweets for token in tweet)
        ngrams = [u for u,c in unigramCount.iteritems() if c >= min_occurrences]
        # bigrams
        bigramCount = Counter()
        trigramCount = Counter()
        for tweet in tweets:
            for a,b,c in zip(tweet[:-1], tweet[1:], tweet[2:]):
                if unigramCount[a] >= min_occurrences and unigramCount[b] >= min_occurrences:
                    bigramCount.update([(a, b)])
                    if unigramCount[c] >= min_occurrences:
                        trigramCount.update([(a, b, c)])
        if len(tweet) > 1 and unigramCount[tweet[-2]] >= min_occurrences and unigramCount[tweet[-1]] >= min_occurrences:
            bigramCount.update([(tweet[-2], tweet[-1])])
        bigrams = []
        for b, c in bigramCount.iteritems():
            if (float(c) - TweetReader.delta) / (unigramCount[b[0]] * unigramCount[b[1]]) > threshold:
                ngrams.append(b[0] + '_' + b[1])
                bigrams.append(b)
        trigrams = []
        for b, c in trigramCount.iteritems():
            if (float(c) - TweetReader.delta) / (unigramCount[b[0]] * unigramCount[b[1]]) > threshold/2 \
                and (float(c) - TweetReader.delta) / (unigramCount[b[1]] * unigramCount[b[2]]) > threshold/2:
                ngrams.append(b[0] + '_' + b[1] + '_' + b[2])
                trigrams.append(b)
        # FIXME: repeat for multigrams
        return ngrams, bigrams, trigrams
