# -*- coding: utf-8 -*-

"""
Load word embeddings from different representations.
"""

import os
import numpy as np
import logging

# local
from word_dictionary import WordDictionary

# ----------------------------------------------------------------------

class Plain(object):

    @classmethod
    def read_vectors(cls, filename):
        """
        Read an embedding from a plain text file with one vector per 
        line, values separated by whitespace.
        """
        with open(filename, 'rb') as file:
            matrix = np.array([[float(value) for value in line.split()]
                               for line in file])
        return matrix

    @classmethod
    def read_vocabulary(cls, filename):
        """
        Read a vocabulary file containing one word per line.
        Return a list of words.
        """
        words = []
        with open(filename, 'rb') as f:
            for line in f:
                word = unicode(line.strip(), 'utf-8')
                if word:
                    words.append(word)
        return words

    @classmethod
    def write_vocabulary(cls, vocab, filename):
        """
        Write a vocabulary to a file containing one word per line.
        """
        with open(filename, 'wb') as f:
            for word in vocab:
                print >> f, word.encode('utf-8')

    @classmethod
    def write_vectors(cls, filename, matrix):
        """
        Write embedding vectors to a plain text file with one vector per 
        line, values separated by whitespace.
        """
        with open(filename, 'wb') as file:
            for row in matrix:
                print >> file, ' '.join(["%f" % x for x in row])

# ----------------------------------------------------------------------

class Senna(object):

    @classmethod
    def read_vocabulary(cls, filename):
        """
        Read the vocabulary file used by SENNA.
        It has one word per line, all lower case except for the special words
        PADDING and UNKNOWN.
    
        """
        return Plain.vocabulary(filename)

# ----------------------------------------------------------------------

class Word2Embeddings(object):

    @classmethod
    def read_vocabulary(cls, filename):
        """
        Read the vocabulary used with word2embeddings.
        It is the same as a plain text vocabulary, except the embeddings for
        the rare/unknown word are the first two items (before any word in the file).
        """
        return Plain.vocabulary(filename, 'polyglot')

    @classmethod
    def read_vectors(cls, filename):
        """
        Load the feature matrix used by word2embeddings.
        """
        import cPickle as pickle
    
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model.get_word_embeddings()

# ----------------------------------------------------------------------

class Word2Vec(object):

    @classmethod
    def load(cls, filename):
        """
        Load words and vectors from a file in word2vec format.
        """
        words = []
        vectors = []
        with open(filename, 'rb') as f:
            len, size = f.readline().split()
            for line in f:
                items = line.split()
                word = unicode(items[0], 'utf-8')
                words.append(word)
                vectors.append([float(x) for x in items[1:]])
        # vectors for the special symbols, not present in words, will be
        # created later
        return np.array(vectors), words

# ----------------------------------------------------------------------

def generate_vectors(num_vectors, num_features, min_value=-0.1, max_value=0.1):
    """
    Generates vectors of real numbers, to be used as word features.
    Vectors are initialized randomly with values in the interval [min_value, max_value]
    :return: a 2-dim numpy array.
    """
    # set the seed for replicability
    #np.random.seed(42)          # DEBUG

    table = np.random.uniform(min_value, max_value, (num_vectors, num_features))
    logging.debug("Generated %d feature vectors with %d features each." %
                  (num_vectors, num_features))
    
    return table
