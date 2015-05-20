# -*- coding: utf-8 -*-
# distutils: language = c++

"""
Classifier exploiting a convolutional neural network.
"""

# standard
import numpy as np
import cPickle as pickle

# local
from network cimport *
from networkconv import ConvolutionalNetwork
from extractors import Converter

# ----------------------------------------------------------------------

cdef class Classifier(object):
    """
    Classifier.
    """
    
    def __init__(self, converter, labels_dict, left_context, right_context,
                 nn, hidden1_size=0, hidden2_size=0):
        """
        :param converter: the Converter object that extracts features and
           converts them to weights.
        :param labels_dict: dictionary of labels.
        :param left_context: size of left context window.
        :param right_context: size of right context window.
        :param nn: network to be used.
        """
        self.converter = converter
        self.labels_dict = labels_dict
        # label list
        self.labels = sorted(labels_dict, key=labels_dict.get)
        self.nn = nn        # dependency injection
        padding_left = converter.get_padding_left()
        padding_right = converter.get_padding_right()
        self.pre_padding = np.array(left_context * [padding_left])
        self.post_padding = np.array(right_context * [padding_right])

    cpdef predict(self, list tokens):
        """
        Classify a list of tokens. 
        
        :param tokens: a list of strings
        :returns: the predicted class label
        """
        cdef np.ndarray[INT_t,ndim=2] converted = self.converter.convert(tokens)
        cdef np.ndarray[INT_t,ndim=2] input = self.converter.lookup(converted)
        cdef np.ndarray[FLOAT_t,ndim=2] scores = self.nn.predict(input)

        return self.labels[np.argmax(scores)]

    def save(self, file):
        """
        Saves the classifier to a file.
        """
        self.nn.save(file)
        pickle.dump(self.labels_dict, file)
        pickle.dump((len(self.pre_padding), len(self.post_padding)), file)
        self.converter.save(file)

    @classmethod
    def load(cls, file):
        """
        Loads the classifier from a file.
        """
        nn = ConvolutionalNetwork.load(file)
        labels_dict = pickle.load(file)
        (left_context, right_context) = pickle.load(file)
        converter = Converter()
        converter.load(file)

        return cls(converter, labels_dict, left_context, right_context, nn=nn)

