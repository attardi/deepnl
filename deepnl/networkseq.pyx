# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: embedsignature=True
# cython: profile=True

"""
A neural network for tagging sequences.
It uses a sliding window of items through the sequence.
"""

import numpy as np
from itertools import izip
import cPickle as pickle
import sys # DEBUG

# local
from math cimport *
from network cimport *

# for decorations
cimport cython

# ----------------------------------------------------------------------

cdef class SeqGradients(Gradients):

    #cdef readonly np.ndarray output
    #cdef readonly np.ndarray transitions

    def __init__(self, int input_size, int hidden_size, int output_size,
                 int seq_len):
        super(SeqGradients, self).__init__(input_size, hidden_size, output_size)
        self.input = np.zeros((seq_len, input_size)) # overrides Gradients
        self.output = np.zeros((seq_len, output_size))
        self.transitions = np.zeros((output_size + 1, output_size))

    def clear(self, int seq_len):
        super(SeqGradients, self).clear()
        self.input.fill(0.0)
        self.output.fill(0.0)
        self.transitions.fill(0.0)

    def addSquare(self, Gradients grads):
        super(SeqGradients, self).addSquare(grads)
        #self.output += grads.output * grads.output # not needed
        self.transitions += grads.transitions * grads.transitions

# ----------------------------------------------------------------------

cdef class SequenceNetwork(Network):
        
    # transitions
    #cdef public np.ndarray transitions
    
    #cdef readonly np.ndarray input_sequence, hidden_sequence, layer2_sequence
    
    def __init__(self, int input_size, int hidden_size, int output_size):
        super(SequenceNetwork, self).__init__(input_size, hidden_size,
                                              output_size)

        # A_i_j score for jumping from tag i to j
        # A_0_i = transitions[-1]
        high = 1.0
        # +1 is due for the initial transition
        self.transitions = np.random.uniform(-high, high, (output_size + 1, output_size))

    
    def _calculate_delta(self, scores):
        """
        Calculates a matrix with the scores for all possible paths at all given
        points (tokens).
        In the returned matrix, delta[i][j] means the sum of all scores 
        ending in token i with tag j (delta_i(j) in eq. 14 in the paper)
        :return: :param scores: updated.
        """
        # See section 3.4.2 of paper:
        # R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu and
        # P. Kuksa.  Natural Language Processing (Almost) from Scratch.
        # Journal of Machine Learning Research, 12:2493-2537, 2011.
        # @see http://ronan.collobert.com/pub/matos/2012_deeplearning_springer.pdf

        # scores[t][k] = ftheta_k,t
        delta = scores
        # logadd for first token. the transition score of the starting tag must be used.
        # it turns out that logadd = log(exp(score)) = score
        # transitions[-1] represents initial transition, A_0,i in paper (mispelled as A_i,0)
        # delta_0(k) = ftheta_k,0 + A_0,i
        delta[0] += self.transitions[-1]
        
        # logadd for the remaining tokens
        # delta_t(k) = ftheta_k,t + logadd_i(delta_t-1(i) + A_i,k)
        #            = ftheta_k,t + log(Sum_i(exp(delta_t-1(i) + A_i,k)))
        transitions = self.transitions[:-1] # A_i,k
        for token in xrange(1, len(delta)):
            # add and sum by columns
            # newaxis allows adding vector to columns:
            logadd = logsumexp2d(delta[token - 1][:,np.newaxis] + transitions, 0)
            #14 logadd = logsumexp2d(delta[token - 1] + transitions.T, 1)
            delta[token] += logadd
            
        return delta

    @cython.boundscheck(False)
    cpdef bool _calculate_gradients_sll(self, np.ndarray[INT_t,ndim=1] tags,
                                       SeqGradients grads,
                                       np.ndarray[FLOAT_t,ndim=2] scores):
        """
        Calculates the output and transition deltas for each token, using
        Sentence Level Likelihood.
        The aim is to minimize the cost:
        C(theta,A) = logadd(scores for all possible paths) - score(correct path)
        
        :param grads: the gradients to be computed, should be initialized to 0.
        :param scores: the scores computed by the network for the whole sequence.
        :returns: True if normal gradient calculation was performed,
            False, if the error was too small and weights update should be
            skipped.
        """
        # ftheta_i,t = network output for i-th tag, at t-th word
        # s = Sum_i(A_tags[i-1],tags[i] + ftheta_i,i), i < len(sentence)   (12)
        cdef float trans, correct_path_score = 0
        cdef int tag, last_tag = self.output_size
        cdef np.ndarray[FLOAT_t,ndim=1] nth_scores
        for tag, nth_scores in izip(tags, scores):
            trans = 0 if self.transitions is None else self.transitions[last_tag, tag]
            correct_path_score += trans + nth_scores[tag]
            last_tag = tag
        
        # delta[t] = delta_t in equation (14)
        cdef np.ndarray[FLOAT_t,ndim=2] delta # (len(sentence), output_size)
        delta = self._calculate_delta(scores)

        # logadd_i(delta_T(i)) = log(Sum_i(exp(delta_T(i))))
        # Sentence-level Log-Likelihood (SLL)
        # C(ftheta,A) = logadd_j(s(x, j, theta, A)) - score(correct path)
        #error = np.log(np.sum(np.exp(delta[-1]))) - correct_path_score
        error = logsumexp(delta[-1]) - correct_path_score
        self.error += error
        
        # if the error is too low, don't bother training (saves time and avoids
        # overfitting). An error of 0.01 means a log-prob of -0.01 for the right
        # tag, i.e., more than 99% probability
        # error 0.69 -> 50% probability for right tag (minimal threshold)
        # error 0.22 -> 80%
        # error 0.1  -> 90%
        if error <= 0.01:
            return False
        
        # things get nasty from here
        
        # compute the gradients for the last token
        # dC_logadd / ddelta_T(i) = e(delta_T(i))/Sum_k(e(delta_T(k)))
        # negative gradients
        grads.output[-1] = -softmax(delta[-1])

        transitions_t = 0 if self.transitions is None else self.transitions[:-1].T
        
        # delta[i][j]: sum of scores of all path that assign tag j to ith-token

        # now compute the gradients for the other tokens, from last to first
        cdef np.ndarray[FLOAT_t,ndim=2] path_scores # (output_size, output_size)
        for t in range(len(scores) - 2, -1, -1):
            
            # sum the scores for all paths ending with each tag i at token t
            # with the transitions from tag i to the next tag j
            # Obtained by transposing twice
            # [delta_t-1(i)+A_j,i]T
            path_scores = (delta[t] + transitions_t).T

            # normalize over all possible tag paths using a softmax,
            # along the columns.
            # softmax is the division of an exponential by the sum of all exponentials
            # (yields a probability)
            # e(delta_t-1(i)+A_i,j) / Sum_k e(delta_t-1(k)+A_k,j)
            path_scores = softmax2d(path_scores)

            # multiply each value in the softmax by the gradient at the next tag
            # dC_logadd / ddelta_t(i) * path_scores
            # Attardi: negative since output[t + 1] already negative
            grad_times_softmax = grads.output[t + 1] * path_scores
            # dC / dA_i,j
            grads.transitions[:-1, :] += grad_times_softmax
            
            # sum all transition gradients by row to find the network gradients
            # Sum_j(dC_logadd / ddelta_t(j) * path_scores(j))
            # Attardi: negative since grad_times_softmax already negative
            grads.output[t] = np.sum(grad_times_softmax, 1)

        # find the gradients for the starting transition
        # there is only one possibility to come from, which is the sentence start
        grads.transitions[-1] = grads.output[0]
        
        # now, add +1 to the correct path
        last_tag = self.output_size
        for token, tag in enumerate(tags):
            grads.output[token][tag] += 1 # negative gradient
            if self.transitions is not None:
                grads.transitions[last_tag][tag] += 1 # negative gradient
            last_tag = tag
        
        return True

    @cython.boundscheck(False)
    cpdef bool _calculate_gradients_wll(self, np.ndarray[INT_t,ndim=1] tags,
                                       SeqGradients grads,
                                       np.ndarray[FLOAT_t,ndim=2] scores):
        """
        Calculates the output for each token, using Word Level Likelihood.
        The aim is to minimize the word-level log-likelihood:
        C(ftheta) = logadd_j(ftheta_j) - ftheta_y,
        where y is the sequence of correct tags
        
        :returns: if True, normal gradient calculation was performed.
            If False, the error was too low and weight correction should be
            skipped.
        """
        # compute the negative gradient with respect to ftheta
        # dC / dftheta_i = e(ftheta_i)/Sum_k(e(ftheta_k))
        cdef np.ndarray[FLOAT_t,ndim=2] exponentials = np.exp(scores)
        # FIXME: use logsumexp
        # ((len(sentence), self.output_size))
        grads.output = -(exponentials.T / exponentials.sum(1)).T

        # correct path and its gradient
        correct_path_score = 0
        token = 0
        for tag, net_scores in izip(tags, scores):
            grads.output[token][tag] += 1 # negative gradient
            token += 1
            correct_path_score += net_scores[tag]

        # C(ftheta) = logadd_j(ftheta_j) - score(correct path)
        #error = np.log(np.sum(np.exp(scores))) - correct_path_score
        error = logsumexp(scores) - correct_path_score
        # approximate
        #error = np.max(scores) - correct_path_score
        self.error += error

        return True

    @cython.boundscheck(False)
    cpdef np.ndarray[INT_t,ndim=1] _viterbi(self,
                                            np.ndarray[FLOAT_t,ndim=2] scores):
        """
        Performs a Viterbi search over the scores for each tag using
        the transitions matrix.
        :return: the most likely sequence of tokens.
            If no matrix was supplied, return the tags with the highest scores individually.
        """
        # pretty straightforward
        if self.transitions is None or len(scores) == 1:
            return scores.argmax(1)

        cdef np.ndarray[FLOAT_t,ndim=2] path_scores = np.empty_like(scores)
        cdef np.ndarray[INT_t,ndim=2] path_backtrack = np.empty_like(scores, np.int)
        
        # now the actual Viterbi algorithm
        # first, get the scores for each tag at token 0
        # the last row of the transitions table has the scores for the first tag
        path_scores[0] = scores[0] + self.transitions[-1]
        
        output_range = np.arange(self.output_size) # outside loop
        transitions = self.transitions[:-1]        # idem

        cdef int i
        for i in xrange(1, len(scores)):
            
            # each line contains the score until each tag t plus the transition to each other tag t'
            prev_score_and_trans = (path_scores[i - 1] + transitions.T).T
            
            # find the previous tag that yielded the max score
            path_backtrack[i] = prev_score_and_trans.argmax(0)
            path_scores[i] = prev_score_and_trans[path_backtrack[i], 
                                                  output_range] + scores[i]
            
        # now find the maximum score for the last token and follow the backtrack
        cdef np.ndarray[INT_t,ndim=1] answer = np.empty(len(scores), dtype=np.int)
        answer[-1] = path_scores[-1].argmax()
        cdef int previous_tag = path_backtrack[-1][answer[-1]]
        
        for i in range(len(scores) - 2, 0, -1):
            answer[i] = previous_tag
            previous_tag = path_backtrack[i][previous_tag]
        
        answer[0] = previous_tag
        return answer

    cdef _backpropagate(self, SeqGradients grads):
        """
        Backpropagate the gradients of the cost.
        :param grads: where to store computed gradients.
        """
        # See Appendix A of paper.

        # f_1 = input_sequence
        # f_2 = M_1 f_1 + b_2 = layer2_sequence
        # f_3 = hardTanh(f_2) = hidden_sequence
        # f_4 = M_2 f_3 + b_4

        # For l = 4..1 do:
        # dC / dtheta_l = df_l / dtheta_l dC / df_l		(19)
        # dC / df_l-1 = df_l / df_l-1 dC / df_l			(20)

        #
        # Compute the gradients of the cost for each layer
        #
        # layer 4: output layer
        # dC / dW_4 = dC / df_4 f_3.T				(22)
        # (len, output_size).T (len, hidden_size) = (output_size, hidden_size)
        grads.output.T.dot(self.hidden_sequence, grads.output_weights)

        # dC / db_4 = dC / df_4					(22)
        # (output_size) += ((len(sentence), output_size))
        # sum by column, i.e. all changes through the sentence
        grads.output.sum(0, out=grads.output_bias)

        # dC / df_3 = M_2.T dC / df_4				(23)
        #  (len, output_size) (output_size, hidden_size) = (len, hidden_size)
        dCdf_3 = grads.output.dot(self.output_weights)

        # layer 3: HardTanh layer
        # no weights to adjust

        # dC / df_2 = hardtanhd(f_2) * dC / df_3
        # (len, hidden_size) * (len, hidden_size) = (len, hidden_size)
        # FIXME: this goes quickly to 0.
        dCdf_2 = hardtanhe2d(self.hidden_sequence) * dCdf_3

        # df_2 / df_1 = M_1

        # layer 2: linear layer
        # dC / dW_2 = dC / df_2 f_1.T				(22)
        # (len, hidden_size).T (len, input_size) = (hidden_size, input_size)
        dCdf_2.T.dot(self.input_sequence, grads.hidden_weights)

        # dC / db_1 = dC / df_2					(22)
        # sum by column contribution by each token
        dCdf_2.sum(0, out=grads.hidden_bias)

        # dC / df_1 = M_1.T dC / df_2
        # (len, hidden_size) (hidden_size, input_size) = (len, input_size)
        dCdf_2.dot(self.hidden_weights, grads.input)
        #print >> sys.stderr, 'hwg', grads.hidden_weights[:4,:4], grads.hidden_weights[-4:,-4:] # DEBUG
        #print >> sys.stderr, 'hbg', grads.hidden_bias[:4], grads.hidden_bias[-4:] # DEBUG
        #print >> sys.stderr, 'ig', grads.input[0,:4], grads.input[-1,-4:] # DEBUG

    cdef _update(self, SeqGradients grads, float learning_rate,
                 SeqGradients ada=None):
        """
        Adjust the weights.
        :param ada: cumulative square gradients for performing AdaGrad.
        """
        super(SequenceNetwork, self).update(grads, learning_rate, ada)

        # Adjusts the transition scores table with the calculated gradients.
        if self.transitions is not None:
            if ada:
                ada.transitions += grads.transitions * grads.transitions
                self.transitions += learning_rate * grads.transitions / np.sqrt(ada.transitions + adaEps)
            else:
                self.transitions += grads.transitions * learning_rate # DEBUG / 10

    def save(self, file):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes and transitions.
        """
        pickle.dump([self.input_size, self.hidden_size, self.output_size,
                     self.hidden_weights, self.hidden_bias,
                     self.output_weights, self.output_bias,
                     self.transitions], file)

    @classmethod
    def load(cls, file):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes, padding and 
        distance tables, as well as all converters' data.
        """
        nn = cls.__new__(cls)

        data = np.load(file)
        nn.input_size, nn.hidden_size, nn.output_size, nn.hidden_weights, \
            nn.hidden_bias, nn.output_weights, nn.output_bias, \
            nn.transitions = data

        return nn
