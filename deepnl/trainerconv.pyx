# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: profile=True

"""
Train a DL Convolutional Neural Network.
"""

import numpy as np

# local
from networkconv cimport *
from trainer cimport Trainer

# for decorations
cimport cython

cdef class ConvTrainer(Trainer):
    """
    Trainer for a convolutional network.
    """

    def __init__(self, Converter converter, float learning_rate,
                 int left_context, int right_context,
                 int hidden1_size, int hidden2_size, labels_dict, verbose=False):
        # sum the number of features in all extractors' tables 
        input_size = (left_context + 1 + right_context) * converter.size()
        nn = ConvolutionalNetwork(input_size, hidden1_size, hidden2_size, len(labels_dict))
        super(ConvTrainer, self).__init__(converter, learning_rate,
                                          left_context, right_context,
                                          nn, verbose=verbose)
    
