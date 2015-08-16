# -*- coding: utf-8 -*-
# distutils: language = c++

"""
A convolutional neural network for NLP tasks.
"""

# standard
import logging
from itertools import izip
import cPickle as pickle
import sys                      # DEBUG

import numpy as np
cimport numpy as np

# local
from math cimport *
from network cimport *

# for decorations
cimport cython

# ----------------------------------------------------------------------

cdef class ConvVariables(Variables):
    """
    Visible and hidden variables.
    Unique to each thread.
    """
    
    #cdef public np.ndarray hidden2, conv

    def __init__(self, feat_size=0, hidden_size=0, hidden2_size=0,
                 output_size=0, slen=0, pool_size=0):
        """
        :param feat_size: number of features.
        :param slen: (padded) sentence length.
        """
        # input variables are all concatenated into a single vector,
        # so that forward propagation can be performed using slices
        super(ConvVariables, self).__init__(feat_size * slen,
                                            hidden_size, output_size)
        self.conv = np.empty((slen - pool_size + 1, hidden_size))
        self.hidden2 = np.empty(hidden2_size)

# ----------------------------------------------------------------------

cdef class ConvParameters(Parameters):
    """
    Network parameters: weights and biases.
    Shared by threads.
    """
    # the second hidden layer
    #cdef public np.ndarray hidden2_weights, hidden2_bias
    
    def __init__(self, int input_size, int hidden1_size, int hidden2_size, int output_size):
        """
        :param input_size: number of input variables, #features x pool size
        """
        #super(ConvParameters, self).__init__(input_size, hidden1_size, output_size)
        self.hidden_weights = np.zeros((hidden1_size, input_size), dtype=float)
        self.hidden_bias = np.zeros(hidden1_size, dtype=float)
        self.hidden2_weights = np.zeros((hidden2_size, hidden1_size), dtype=float)
        self.hidden2_bias = np.zeros(hidden2_size, dtype=float)
        self.output_weights = np.zeros((output_size, hidden2_size), dtype=float)
        self.output_bias = np.zeros(output_size, dtype=float)

    def initialize(self, int input_size, int hidden_size, int hidden2_size,
                   int output_size):
        #super(ConvParameters, self).initialize(input_size, hidden_size, output_size)
        high = 2.38 / np.sqrt(input_size) # [Bottou-88]
        self.hidden_weights = np.random.uniform(-high, high, (hidden_size, input_size))
        self.hidden_bias = np.random.uniform(-high, high, (hidden_size))
        high = 2.38 / np.sqrt(hidden_size) # [Bottou-88]
        self.hidden2_weights = np.random.uniform(-high, high, (hidden2_size, hidden_size))
        high = 2.38 / np.sqrt(hidden2_size) # [Bottou-88]
        self.hidden2_bias = np.random.uniform(-high, high, hidden2_size)
        self.output_weights = np.random.uniform(-high, high, (output_size, hidden2_size))
        high = 2.38 / np.sqrt(output_size) # [Bottou-88]
        self.output_bias = np.random.uniform(-high, high, output_size)

    cpdef update(self, Gradients grads, float learning_rate,
                 Gradients ada=None):
        super(ConvParameters, self).update(grads, learning_rate, ada)
        if ada:
            global adaEps
            self.hidden2_weights += learning_rate * grads.hidden2_weights / np.sqrt(ada.hidden2_weights + adaEps)
            self.hidden2_bias += learning_rate * grads.hidden2_bias / np.sqrt(ada.hidden2_bias + adaEps)
        else:
            # divide by the fan-in
            self.hidden2_weights += grads.hidden2_weights * learning_rate / self.hidden_size
            self.hidden2_bias += grads.hidden2_bias * learning_rate / self.hidden_size

    def save(self, file):
        """
        Saves the parameters to a file.
        It will save the weights and biases.
        """
        pickle.dump([self.hidden_weights, self.hidden_bias,
                     self.output_weights, self.output_bias,
                     self.hidden2_weights, self.hidden2_bias], file)

    @classmethod
    def load(cls, file):
        """
        Loads the parameters from a file.
        It will load weights and biases.
        """
        p = cls.__new__(cls)

        data = pickle.load(file)
        p.hidden_weights, p.hidden_bias, \
            p.output_weights, p.output_bias, \
            p.hidden2_weights, p.hidden2_bias = data
        return p

# ----------------------------------------------------------------------

cdef class ConvGradients(Gradients):

    # cdef readonly int hidden2_size
    # cdef public np.ndarray conv
    # input has size input_size * (slen + pool_size), pool_size is for padding

    def __init__(self, int feat_size, int hidden1_size, int hidden2_size,
                 int output_size, int pool_size, int slen):
        """
        :param feat_size: number of features
        """
        #super(ConvGradients, self).__init__(feat_size * pool_size, hidden1_size, output_size)
        input_size = feat_size * pool_size
        self.hidden_weights = np.zeros((hidden1_size, input_size), dtype=float)
        self.hidden_bias = np.zeros(hidden1_size, dtype=float)
        self.hidden2_weights = np.zeros((hidden2_size, hidden1_size), dtype=float)
        self.hidden2_bias = np.zeros(hidden2_size, dtype=float)
        self.output_weights = np.zeros((output_size, hidden2_size), dtype=float)
        self.output_bias = np.zeros(output_size, dtype=float)

        self.input = np.zeros(feat_size * (slen + pool_size), dtype=float)
        self.conv = np.empty((slen, hidden1_size))

    def clear(self):
        super(ConvGradients, self).clear()
        self.hidden2_weights.fill(0.0)
        self.hidden2_bias.fill(0.0)
        self.conv.fill(0.0)

    def addSquare(self, Gradients grads):
        """For adaGrad"""
        super(ConvGradients, self).addSquare(grads)
        self.hidden2_weights += grads.hidden2_weights * grads.hidden2_weights
        self.hidden2_bias += grads.hidden2_bias * grads.hidden2_bias

# ----------------------------------------------------------------------

cdef class ConvolutionalNetwork(Network):
    
    def __init__(self, int input_size, int hidden1_size, int hidden2_size,
                 int output_size, int pool_size):
        """
        Creates a new convolutional neural network.
        :parameter input_size: the number of network input variables.
        """
        super(ConvolutionalNetwork, self).__init__(input_size,
                                                   hidden1_size, output_size)

        self.pool_size = pool_size
        self.hidden2_size = hidden2_size
        self.p = ConvParameters(input_size, hidden1_size,
                                hidden2_size, output_size)
        self.p.initialize(input_size, hidden1_size, hidden2_size,
                          output_size)
        
    def description(self):
        """Returns a textual description of the network."""
        # table_dims = [str(t.shape[1]) for t in self.feature_tables]
        # table_dims =  ', '.join(table_dims)
        
        desc = """
Input layer size: %d
Convolution layer size: %d 
Second hidden layer size: %d
Output size: %d
""" % (self.input_size, self.hidden_size, self.hidden2_size, self.output_size)
        
        return desc
    
    cdef variables(self, int slen=1):
        """
        Allocate variables.
        :param slen: sentence length.
        """
        return ConvVariables(self.input_size / self.pool_size,
                             self.hidden_size, self.hidden2_size,
                             self.output_size, slen, self.pool_size)

    cdef gradients(self, int slen=1):
        """Allocate gradients.
        :param slen: sentence length.
        """
        return ConvGradients(self.input_size / self.pool_size, self.hidden_size,
                             self.hidden2_size, self.output_size,
                             self.pool_size, slen)

    cpdef forward(self, Variables vars):
        """Runs the network using the given :param vars:"""

        # convolution layer
        # f_1 = W_1 * f_0 + b1
        cdef int slen = len(vars.conv)
        cdef int feat_size = self.input_size / self.pool_size
        cdef int i, s = 0
        cdef ConvParameters p = self.p
        # run through all windows in the sentence
        # vars.input contains inputs for the whole sentence
        for i in xrange(slen):
            # (hidden_size, input_size) * input_size = hidden_size
            p.hidden_weights.dot(vars.input[s:s+self.input_size], vars.conv[i])
            vars.conv[i] += p.hidden_bias
            s += feat_size
        # max over time layer
        # f_2 = max(f_1)
        vars.hidden[:] = vars.conv.max(0)  # max by columns
        # first linear layer
        # f_3 = W_2 * f_2 + b_2
        p.hidden2_weights.dot(vars.hidden, vars.hidden2)
        vars.hidden2 += p.hidden2_bias
        # activation
        # f_4 = hardtanh(f_3)
        hardtanh(vars.hidden2, vars.hidden2)
        # second linear layer
        # f_5 = W_3 * f_4 + b_3
        p.output_weights.dot(vars.hidden2, vars.output)
        vars.output += p.output_bias

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[FLOAT_t,ndim=1] predict(self, list tokens):
        """
        Runs the network for each element in the sentence and returns 
        the predicted probabilities for each possible outcome.
        
        :param tokens: a list of strings.
        :return: the predicted scores for each possible output variable.
        """

        cdef np.ndarray[INT_t,ndim=2] converted = self.converter.convert(tokens)
        # add padding to the sentence
        cdef np.ndarray[INT_t,ndim=2] padded_sentence = \
            np.concatenate((self.pre_padding,
                            converted,
                            self.post_padding))

        # allocate variables
        vars = self.variables(len(padded_sentence))
        # lookup layer
        self.converter.lookup(padded_sentence, vars.input)

        self.forward(vars)
        return vars.output

    cdef float backpropagate(self, int y, Variables vars, Gradients grads):
        """
        Cost is the hinge loss.
        Compute the gradients of the cost for each layer.
        :param y: the correct outcome.
        :param vars: the network variables.
        :param grads: were to store the gradients.
        :return: the hinge loss.
        """

        # multiclass hinge loss
        # hl(x, y) = max(0, 1 + max_t!=y f(x)[t] - f(x)[y])
        # Hinge loss is 0 if the score of the correct label exceeds the score
        # of every other label by by a margin of at least 1.
        # m = argmax_t!=y f(x)[t]
        # dhl / df [y] = -1 if f(x)[m] - f(x)[y] > 1, else 0
        # dhl / df [t] = +1 if f(x)[t] - f(x)[y] > 1, else 0
        cdef int m = np.argmax(vars.output)
        cdef float fx_m = vars.output[m]
        cdef float fx_y = vars.output[y]
        cdef float hinge_loss = max(0.0, 1 + fx_m - fx_y)

        if hinge_loss == 0.0:
            return hinge_loss
        cdef ConvParameters p = self.p

        # minimizing C(f_5) = hl(f_5)
        # f_5 = W_3 f_4 + b_3
        # dC / db_4 = dC / df_5					(22)
        # negative gradient:
        grads.output_bias[:] = np.where(vars.output - fx_y > 1, -1, 0) # -1
        grads.output_bias[y] = fx_y - fx_m < 1 # +1
        # dC / dW_4 = dC / df_5 * f_4				(22)
        # (output_size) x (hidden2_size) = (output_size, hidden2_size)
        np.outer(grads.output_bias, vars.hidden2, grads.output_weights)
        # dC / df_4 = dC / df_5 * W_4				(23)
        # (output_size) * (output_size, hidden2_size) = (hidden2_size)
        grads.output_bias.dot(p.output_weights, grads.hidden2_bias)

        # f_4 = hardtanh(f_3)
        # dC / df_3 = dC / df_4 * hardtanhd(f_4)
        hardtanhe(vars.hidden2, vars.hidden2)
        grads.hidden2_bias *= vars.hidden2

        # f_3 = W_2 f_2 + b_2
        # dC / db_2 = dC / df_3					(22)

        # dC / dW_2 = dC / df_3 * f_2				(22)
        # (hidden2_size) x (hidden_size) = (hidden2_size, hidden_size)
        try:
            np.outer(grads.hidden2_bias, vars.hidden, grads.hidden2_weights)
        except Exception, e:
            print >> sys.stderr, e, np.abs(vars.hidden).min(), np.abs(grads.hidden2_weights).min() # DEBUG

        # dC / df_2 = dC / df_3 * W_2				(23)
        # (hidden2_size) * (hidden2_size, hidden_size) = (hidden_size)
        grads.hidden2_bias.dot(p.hidden2_weights, grads.hidden_bias)

        # Max layer
        # f_2 = max(f_1)
        # dC / df_1 = dC / df_2 * maxd(f_1)  
        grads.conv.fill(0.0)

        # (_size)
        cdef np.ndarray a = vars.conv.argmax(1) # indices of max values

        # @see Appendix A.4:
        # (dC / df_1)[t,i] = (dC / df_2)[t,i] if t = a[i], else 0
        cdef int i, ai
        for i, ai in enumerate(a):
            grads.conv[i, ai] = grads.hidden_bias[ai]

        # Convolution layer
        cdef int slen = len(vars.conv)
        cdef int feat_size = self.input_size / self.pool_size

        # f_1 = [W_1 f_0[t:t+w] + b_1   for t < slen]
        # see Appendix A.3:
        # dC / db_1 = Sum_t((dC / df_1)[t])
        # dC / dW_1 = Sum_t((dC / df_1)[t] * f_0[t:t+w])
        # (dC / df_1)[t:t+w] += (dC / df_2)[t] * W_1
        grads.hidden_weights.fill(0.0)
        grads.hidden_bias.fill(0.0)
        grads.input.fill(0.0)
        cdef int s = 0
        for t in xrange(slen):
            # dC / db_1 = Sum_t((dC / df_1)[t])
            grads.hidden_bias += grads.conv[t]

            # dC / dW_1 = Sum_t((dC / df_1)[t] * f_0[t:t+w])
            # (hidden_size) * (input_size) = (hidden_size, input_size)
            grads.hidden_weights += np.outer(grads.conv[t], vars.input[s:s+self.input_size])

            # (dC / df_1)[t:t+w] += (dC / df_2)[t] * W_1
            # (hidden_size) * (hidden_size, input_size) = (input_size)
            grads.input[s:s+self.input_size] += grads.conv[t].dot(p.hidden_weights)
            s += feat_size

        return hinge_loss

    @classmethod
    def load(cls, file):
        """
        Loads the neural network from a file.
        It will load weights and biases.
        """
        nn = cls.__new__(cls)
        nn.p = ConvParameters.load(file)
        nn.input_size = nn.p.hidden_weights.shape[1]
        nn.hidden_size = nn.p.hidden_weights.shape[0]
        nn.hidden2_size = nn.p.hidden2_weights.shape[1]
        nn.output_size = nn.p.output_weights.shape[0]
        return nn
