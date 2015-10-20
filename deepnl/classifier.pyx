# -*- coding: utf-8 -*-
# distutils: language = c++

"""
Classifier exploiting a neural network.
"""

# standard
import numpy as np
from numpy import int32 as INT
cimport numpy as np
import cPickle as pickle

# local
from network cimport *
from extractors import Converter
from extractors cimport Converter
from utils import import_class

# ----------------------------------------------------------------------

cdef class Classifier(object):
    """
    Classifier using a neural network.
    """

    cdef Converter converter
    cdef list labels
    cdef readonly Network nn
    cdef np.ndarray pre_padding, post_padding

    def __init__(self, Converter converter, list labels, int_t left_context, int_t right_context,
                 Network nn):
        """
        :param converter: the Converter object that extracts features and
           converts them to weights.
        :param labels: list of labels.
        :param left_context: size of left context window.
        :param right_context: size of right context window.
        :param nn: network to be used.
        """
        self.converter = converter
        self.labels = labels
        self.nn = nn        # dependency injection
        cdef np.ndarray[int_t] padding_left = converter.get_padding_left()
        cdef np.ndarray[int_t] padding_right = converter.get_padding_right()
        self.pre_padding = np.array(left_context * [padding_left], dtype=INT)
        self.post_padding = np.array(right_context * [padding_right], dtype=INT)

    cpdef predict(self, list tokens):
        """
        Classify a list of tokens. 
        
        :param tokens: a list of tokens, each a list of attributes.
        :returns: the predicted class label
        """
        cdef np.ndarray[int_t,ndim=2] converted = self.converter.convert(tokens)
        # add padding to the sentence
        cdef np.ndarray[int_t,ndim=2] padded_sentence = \
            np.concatenate((self.pre_padding, converted, self.post_padding))

        # allocate variables
        vars = self.nn.variables(len(padded_sentence))
        # lookup layer
        self.converter.lookup(padded_sentence, vars.input)
        output = self.nn.forward(vars)
        return self.labels[np.argmax(output)]

    def save(self, file):
        """
        Saves the classifier to a file.
        """
        netClass = type(self.nn) # fully qualified name
        pickle.dump(netClass.__module__+'.'+netClass.__name__, file)
        self.nn.save(file)
        pickle.dump(self.labels, file)
        pickle.dump((len(self.pre_padding), len(self.post_padding)), file)
        self.converter.save(file)

    @classmethod
    def load(cls, file):
        """
        Loads the classifier from a file.
        """
        classname = pickle.load(file)
        klass = import_class(classname)
        nn = klass.load(file)
        labels = pickle.load(file)
        (left_context, right_context) = pickle.load(file)
        converter = Converter()
        converter.load(file)

        return cls(converter, labels, left_context, right_context, nn=nn)
