# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: embedsignature=True
# cython: profile=True

"""
A neural network for NLP tagging tasks.
"""

# standard
import logging
import sys                      # DEBUG

import numpy as np
cimport numpy as np

# for decorations
cimport cython

# local
from math cimport *

#  Epsilon value for numerical stability in AdaGrad's division
cdef float adaEps = 1e-6

# ----------------------------------------------------------------------

cdef class Variables(object):
    
    #cdef public np.ndarray input, hidden, output

    def __init__(self, input_size=0, hidden_size=0, output_size=0):
        self.input = np.empty(input_size) if input_size else None
        self.hidden = np.empty(hidden_size) if hidden_size else None
        self.output = np.empty(output_size) if output_size else None

# ----------------------------------------------------------------------

cdef class Parameters(object):
    """
    Network parameters: weights and biases.
    Parameters are shared among threads in ASGD.
    """
    
    def __init__(self, int input_size, int hidden_size, int output_size):
        self.output_weights = np.zeros((output_size, hidden_size), dtype=float)
        self.output_bias = np.zeros(output_size, dtype=float)
        self.hidden_weights = np.zeros((hidden_size, input_size), dtype=float)
        self.hidden_bias = np.zeros(hidden_size, dtype=float)

# ----------------------------------------------------------------------

cdef class Gradients(Parameters):
    """
    Gradients for all network Parameters, plus input gradients.
    """
    
    def __init__(self, int input_size, int hidden_size, int output_size):
        super(Gradients, self).__init__(input_size, hidden_size, output_size)
        self.input = np.zeros(input_size, dtype=float)

    def clear(self):
        self.output_weights.fill(0.0)
        self.output_bias.fill(0.0)
        self.hidden_weights.fill(0.0)
        self.hidden_bias.fill(0.0)
        self.input.fill(0.0)

    def addSquare(self, Gradients grads):
        self.output_weights += grads.output_weights * grads.output_weights
        self.output_bias += grads.output_bias * grads.output_bias
        self.hidden_weights += grads.hidden_weights * grads.hidden_weights
        self.hidden_bias += grads.hidden_bias * grads.hidden_bias

# ----------------------------------------------------------------------

cdef class Network(Parameters):

    def __init__(self, int input_size, int hidden_size, int output_size):
        """
        :param left_context: size of left context window.
        :param right_context: size of right context window.
        :param converter: the Converter object that extracts features and
        converts them to weights.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Note : optimal initialization of weights may depend on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid()
        #        compared to tanh().

        # creates the weight matrices

        # set the seed for replicability
        #np.random.seed(42)      # DEBUG

        high = 2.38 / np.sqrt(input_size) # [Bottou-88]
        #high = 2.45 / np.sqrt(input_size + hidden_size) # Al-Rfou
        self.hidden_weights = np.random.uniform(-high, high, (hidden_size, input_size))
        self.hidden_bias = np.random.uniform(-high, high, (hidden_size))
        #self.hidden_bias = np.zeros(hidden_size, dtype=float) # Al-Rfou

        high = 2.38 / np.sqrt(hidden_size) # [Bottou-88]
        #high = 2.45 / np.sqrt(hidden_size + output_size) # Al-Rfou
        self.output_weights = np.random.uniform(-high, high, (output_size, hidden_size))
        self.output_bias = np.random.uniform(-high, high, (output_size))
        #self.output_bias = np.zeros(output_size, dtype=float) # Al-Rfou

        # saver fuction
        self.saver = lambda nn: None
    
    def description(self):
        """
        Returns a textual description of the network.
        """
        desc = """
        Input layer size: %d
        Hidden layer size: %d
        Output size: %d
        """ % (self.input_size, self.hidden_size, self.output_size)
        
        return desc
    
    cdef variables(self, int slen=0):
        """Allocate variables."""
        return Variables(self.input_size, self.hidden_size, self.output_size)

    cdef gradients(self, int slen=0):
        """Allocate variables."""
        return Gradients(self.input_size, self.hidden_size, self.output_size)

    cpdef run(self, Variables vars):
        """
        Runs the network on the given variables: hidden and visible. 
        """
        # (hidden_size, input_size) . input_size = hidden_size
        self.hidden_weights.dot(vars.input, vars.hidden)
        vars.hidden += self.hidden_bias
        hardtanh(vars.hidden, vars.hidden)
        self.output_weights.dot(vars.hidden, vars.output)
        vars.output += self.output_bias

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

        if hinge_loss == 0.0:
            return hinge_loss
        # f_4 = W_2 f_3 + b_2
        # dC / db_2 = dC / df_4
        np.copyto(grads.output_bias, vars.output)
        grads.output_bias[y] += 1.0 # negative gradient
        # dC / dW_2 = dC / df_4 f_3
        # output_size x hidden_size
        np.outer(grads.output_bias, vars.hidden, grads.output_weights)
        # dC / df_3 = dC / df_4 * W_2				(23)

        # f_3 = hardtanh(f_2)
        # dC / df_2 = dC / df_3 * hardtanhd(f_2)
        hardtanhe(vars.hidden, vars.hidden)

        # f_2 = W_1 f_1 + b_1
        # dC / db_1 = dC / df_2					(22)
        # (output_size) * (output_size x hidden_size) = (hidden_size)
        # grads.hidden_bias = vars.hidden * grads.output_bias.dot(self.output_weights)
        # dC / df_3
        grads.output_bias.dot(self.output_weights, grads.hidden_bias)
        #           * hardtanh(f_2) = dC / df_2
        grads.hidden_bias *= vars.hidden

        # dC / dW_1 = dC / df_2 * f_1
        # (hidden_size) x (input_size) = (hidden_size, input_size)
        np.outer(grads.hidden_bias, vars.input, grads.hidden_weights)

        # dC / df_1 = dC / df_2 * W_1				(23)

        # Lookup layer
        # f_1 = W_0 f_0
        # dC / dW_0 = dC / df_1 * W_0
        # (hidden_size) * (hidden_size, input_size) = (input_size)
        grads.hidden_bias.dot(self.hidden_weights, grads.input)

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
        if ada:
            ada.addSquare(grads)
            self.output_weights += learning_rate * grads.output_weights / np.sqrt(ada.output_weights + adaEps)
            self.output_bias += learning_rate * grads.output_bias / np.sqrt(ada.output_bias + adaEps)
            self.hidden_weights += learning_rate * grads.hidden_weights / np.sqrt(ada.hidden_weights + adaEps)
            self.hidden_bias += learning_rate * grads.hidden_bias / np.sqrt(ada.hidden_bias + adaEps)
        else:
            # divide by the fan-in
            self.output_weights += grads.output_weights * learning_rate / 100 # DEBUG / self.hidden_size
            self.output_bias += grads.output_bias * learning_rate / 100 # DEBUG self.hidden_size
            #print >> sys.stderr, 'uhw', learning_rate, self.hidden_weights[:2,:2], grads.hidden_weights[:2,:2] # DEBUG
            self.hidden_weights += grads.hidden_weights * learning_rate / 100 # DEBUG self.input_size
            self.hidden_bias += grads.hidden_bias * learning_rate / 100 # DEBUG self.input_size

    def save(self, file):
        """
        Saves the neural network to a file.
        It saves the weights and biases.
        """
        np.savez(file, hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_bias=self.hidden_bias, output_bias=self.output_bias)

    @classmethod
    def load(cls, file):
        """
        Loads the neural network from a file.
        It loads weights, biases and sizes.
        """
        data = np.load(file)
        
        nn = Network.__new__(cls)
        nn.hidden_weights = data['hidden_weights']
        nn.hidden_bias = data['hidden_bias']
        nn.output_weights = data['output_weights']
        nn.output_bias = data['output_bias']
        nn.input_size = nn.hidden_weights.shape[1]
        nn.hidden_size = nn.hidden_weights.shape[0]
        nn.output_size = nn.output_weights.shape[0]

        return nn

