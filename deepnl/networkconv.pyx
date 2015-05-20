# -*- coding: utf-8 -*-
# distutils: language = c++

"""
A convolutional neural network for NLP tasks.
"""

import logging
from itertools import izip

from math import *
from network cimport *

# for decorations
cimport cython

# ----------------------------------------------------------------------

cdef class ConvVariables(Variables):
    """Visible and hidden variables.
    Unique to thread"""
    
    #cdef public np.ndarray hidden2, conv, tmax

    def __init__(self, input_size=0, hidden_size=0, hidden2_size=0,
                 output_size=0, slen=0):
        """
        :param input_size: shoule be feature size * window size
        :param slen: sentence length.
        """
        super(ConvVariables, self).__init__(input_size, hidden_size, output_size)
        self.conv = np.empty((slen, hidden_size)) if hidden_size else None
        self.hidden2 = np.empty(hidden2_size) if hidden2_size else None

# ----------------------------------------------------------------------

cdef class ConvParameters(Parameters):
    """
    Network parameters: weights and biases.
    Shared by threads.
    """
    
    def __init__(self, int input_size, int hidden1_size, int hidden2_size, int output_size):
        super(ConvParameters, self).__init__(input_size, hidden1_size, output_size)
        self.hidden2_weights = np.zeros((hidden2_size, hidden1_size), dtype=float)
        self.hidden2_bias = np.zeros(hidden2_size, dtype=float)

# ----------------------------------------------------------------------

cdef class ConvGradients(ConvParameters):

    # cdef public np.ndarray conv

    def __init__(self, int input_size, int hidden1_size, int hidden2_size, int output_size, int slen):
        super(ConvGradients, self).__init__(input_size, hidden1_size, output_size)
        self.hidden2 = np.zeros((hidden2_size, hidden1_size))
        self.bias2 = np.zeros(hidden2_size)
        # temporary
        self.conv = np.empty((slen, hidden_size))

    def clear(self):
        super(ConvGradients, self).clear()
        self.hidden2.fill(0.0)
        self.bias2.fill(0.0)

    def addSquare(self, Gradients grads):
        super(ConvGradients, self).addSquare(grads)
        self.hidden2 += grads.hidden2 * grads.hidden2

# ----------------------------------------------------------------------

cdef class ConvolutionalNetwork(Network):
    
    def __init__(self, int input_size, int hidden1_size, int hidden2_size,
                 int output_size):
        """Creates a new convolutional neural network."""
        super(ConvolutionalNetwork, self).__init__(input_size, hidden1_size,
                                                   output_size)

        self.hidden2_size = hidden2_size
        
        if hidden2_size > 0:
            high = 2.38 / np.sqrt(hidden1_size) # [Bottou-88]
            self.hidden2_weights = np.random.uniform(-high, high, (hidden2_size, hidden1_size))
            high = 2.38 / np.sqrt(hidden2_size) # [Bottou-88]
            self.hidden2_bias = np.random.uniform(-high, high, hidden2_size)
            output_dim = (output_size, hidden2_size)
        else:
            self.hidden2_weights = None
            self.hidden2_bias = None
            output_dim = (output_size, hidden1_size)

        high = 2.38 / np.sqrt(output_dim[1]) # [Bottou-88]
        self.output_weights = np.random.uniform(-high, high, output_dim)
        high = 2.38 / np.sqrt(output_size) # [Bottou-88]
        self.output_bias = np.random.uniform(-high, high, output_size)
        
    def description(self):
        """Returns a textual description of the network."""
        hidden2_size = self.hidden2_size if self.hidden2_weights else 0
        table_dims = [str(t.shape[1]) for t in self.feature_tables]
        table_dims =  ', '.join(table_dims)
        
        desc = """
Input layer size: %d
Convolution layer size: %d 
Second hidden layer size: %d
Output size: %d
""" % (self.input_size, self.hidden_size, hidden2_size, self.output_size)
        
        return desc
    
    cdef variables(self, int slen=0):
        """Allocate variables.
        :param slen: sentence length.
        """
        d = self.converter.size() # num features
        cdef int window_size = len(self.pre_padding) + 1 + len(self.post_padding)
        input_size = d * window_size
        return ConvVariables(input_size, self.hidden_size,
                             self.hidden2_size, self.output_size, slen)

    cdef gradients(self, int slen=0):
        """Allocate gradients.
        :param slen: sentence length.
        """
        d = self.converter.size() # num features
        cdef int window_size = len(self.pre_padding) + 1 + len(self.post_padding)
        input_size = d * window_size
        return ConvGradients(input_size, self.hidden_size,
                             self.hidden2_size, self.output_size, slen)

    cpdef run(self, Variables vars):
        """Runs the network using the given :param vars:"""

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[FLOAT_t,ndim=1] predict(self,
                                            np.ndarray[INT_t,ndim=2] sentence,
                                            vars,
                                            bool train=False):
        """
        Runs the network for each element in the sentence and returns 
        the predicted probabilities for each possible outcome.
        
        :param sentence: a 2-dim numpy array, where each item encodes a token.
        :param train: whether training or predict.
        """

        cdef slen = len(sentence)
        cdef np.ndarray[FLOAT_t,ndim=1] score = np.empty(nn.output_size)
        
        # add padding to the sentence
        cdef np.ndarray[INT_t,ndim=2] padded_sentence = \
            np.concatenate((self.pre_padding,
                            sentence,
                            self.post_padding))

        # allocate variables
        vars = self.variables(slen)
        # lookup layer
        self.converter.lookup(padded_sentence, vars.input)

        # convolution layer
        cdef int i
        # run through all windows in the sentence
        for i in xrange(slen):
            s = i * self.input_size
            # (hidden_size, input_size) . input_size = hidden_size
            self.hidden_weights.dot(vars.input[s:s+self.input_size], vars.conv[i])
            vars.conv[i] += self.hidden_bias
        # max layer
        if train:
            vars.tmax = vars.conv.argmax(0) # indices of max values
        vars.hidden = vars.conv.max(0)  # max by columns
        # first linear layer
        self.hidden2_weights.dot(vars.hidden, vars.hidden2)
        vars.hidden2 += self.hidden2_bias
        # activation
        hardtanh(vars.hidden2, vars.hidden2)
        # second linear layer
        self.output_weights.dot(vars.hidden2, vars.output)
        vars.output += self.output_bias
        
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

        # hinge loss
        cdef float hinge_loss = max(0.0, 1.0 - vars.output[y])
        # alternatively, use negative log-likelihood:
        # nll(x) = -log(P(y|x)) = -log(softmax(f(x))[y])

        if hinge_loss == 0.0:
            return hinge_loss

        # f_5 = W_4 f_4 + b_4
        # dC / db_4 = dC / df_5
        np.copyto(grads.output_bias, vars.output)
        grads.output_bias[y] += 1.0 # negative gradient
        # dC / dW_4 = dC / df_5 * f_4
        # output_size x hidden_size
        np.outer(grads.output_bias, vars.hidden2, grads.output_weights)
        # dC / df_4 = dC / df_5 * W_3				(23)

        # f_4 = hardtanh(f_3)
        # dC / df_3 = dC / df_4 * hardtanhd(f_3)
        hardtanhe(vars.hidden2, vars.hidden2)
        # W_3, b_3 do not exist

        # f_3 = W_2 f_2 + b_2
        # dC / db_2 = dC / df_3					(22)
        # (output_size) * (output_size x hidden2_size) = (hidden2_size)
        # grads.hidden_bias = vars.hidden * grads.output_bias.dot(self.output_weights)
        # dC / df_4
        grads.output_bias.dot(self.output_weights, grads.hidden2_bias)
        #           * hardtanh(f_3) = dC / df_3
        grads.hidden2_bias *= vars.hidden2

        # Max layer
        # f_3 = max(f_2)
        # dC / df_2 = maxd(f_2)
        grads.conv.fill(0.0)
        cdef int i, ai
        for i, ai in enumerate(tmax):
            grads.conv[i, ai] = vars.conv[i, ai]

        # Convolution layer
        # f_2 = vstack(W_1 f_1[t:t+w] + b_1)   for t < slen
        # dC / dW_1 = Sum(dC / df_2[t] * f_1[t:t+w])
        grads.hidden_weights.fill(0.0)
        grads.hidden_bias.fill(0.0)
        grads.input.fill(0.0)
        for t in xrange(slen):
            s = t * self.input_size
            # (hidden_size, input_size) = (hidden_size) * (input_size)
            grads.hidden_weights += np.outer(grads.conv[t], vars.input[s:s+self.input_size])

        # dC / db_1 = Sum(dC / f_2[t])
            # (hidden_size) = (input_size/w) ??
            grads.hidden_bias += vars.input[s:s+self.input_size]

        # dC / df_1[t:t+w] += dC / df_2[t] * W_1
            # (input_size) = (hidden_size) * (hidden_size, input_size)
            grads.input[s:s+self.input_size] += grads.conv[t].dot(vars.hidden_weights)

        return hinge_loss

    cpdef update(self, Gradients grads, float learning_rate,
                 Gradients ada=None):
        """
        Adjust the weights.
        :param ada: cumulative square gradients for performing AdaGrad.
        AdaGrad: G_t, where G(i,i)_t = G(i,i)_t-1 + grad(i)^2
        * i.e. we cumulate the square of gradients in G for parameter p:
        * G += g^2
        * p -= LR * g / sqrt(G + eps)

        Consider using AdaDelta instead:
        http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

        """
        super(ConvolutionalNetwork, self).update(grads, learning_rate, ada)
        if ada:
            self.hidden2_weights += learning_rate * grads.hidden2_weights / np.sqrt(ada.hidden2_weights + adaEps)
            self.hidden2_bias += learning_rate * grads.hidden2_bias / np.sqrt(ada.hidden2_bias + adaEps)
        else:
            # divide by the fan-in
            self.hidden2_weights += grads.hidden2_weights * learning_rate / self.hidden_size
            self.hidden2_bias += grads.hidden2_bias * learning_rate / self.hidden_size

    def save(self, filename):
        """
        Saves the neural network to a file.
        It will save the weights and biases.
        """
        pickle.dump([self.input_size, self.hidden_size, self.hidden2_size,
                     self.output_size,
                     self.hidden_weights, self.hidden_bias,
                     self.output_weights, self.output_bias,
                     self.hidden2_weights, self.hidden2_bias], file)

    @classmethod
    def load(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights and biases.
        """
        nn = cls.__new__(cls)

        data = np.load(file)
        nn.input_size, nn.hidden_size, nn.hidden2_size, nn.output_size, \
            nn.hidden_weights, nn.hidden_bias, \
            nn.output_weights, nn.output_bias, \
            nn.hidden2_weights, nn.hidden2_bias = data
        return nn
    
