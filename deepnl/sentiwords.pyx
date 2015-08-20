# -*- coding: utf-8 -*-
# distutils: language = c++

"""
Train a sentiment specific language model.
"""

import numpy as np
import sys                      # DEBUG

# for decorations
cimport cython

# local
from math cimport *
from words cimport *
from extractors cimport Iterable

# ----------------------------------------------------------------------

cdef class SentGradients(Gradients):

    cdef public np.ndarray hidden_pos
    cdef public np.ndarray hidden_neg
    #cdef public np.ndarray input_pos: as Gradients.input
    cdef public np.ndarray input_neg
    
    def __init__(self, int input_size, int hidden_size, int output_size):
        super(SentGradients, self).__init__(input_size, hidden_size, output_size)
        # gradients for positive hidden variables
        self.hidden_pos = np.zeros(hidden_size, dtype=float)
        # gradients for negative hidden variables
        self.hidden_neg = np.zeros(hidden_size, dtype=float)
        # gradients for negative input variables
        self.input_neg = np.zeros(input_size, dtype=float)

    def clear(self):
        super(SentGradients, self).clear()
        self.hidden_pos.fill(0.0)
        self.hidden_neg.fill(0.0)
        self.input_neg.fill(0.0)

# ----------------------------------------------------------------------

def itertrie(trie, sent, start, depth=0):
    """iterate through all ngrams that occur in :param sent: starting at
    position :param start:"""
    yield 1                     # unigram
    tr = trie
    for cur in xrange(start, len(sent)):
        tok = sent[cur][0]
        if tok in tr:         # part of ngram
            tr = tr[tok]
        else:
            break
    if cur > start+1:
        yield cur-start         # ngram

# ----------------------------------------------------------------------

cdef class SentimentTrainer(LmTrainer): 
    """
    A neural network for sentiment specific language model, aimed 
    at inducing sentiment-specific word representations.
    @see Tang et al. 2014. Learning Sentiment-SpecificWord Embedding for Twitter Sentiment Classification.
    http://aclweb.org/anthology/P14-1146
    """
    
    # polarities of each tweet
    cdef list polarities

    # alpha parameter: relative weight of standard and sentiment errors.
    cdef double alpha

    cdef np.ndarray input_adagrads_neg
    cdef np.ndarray input_adagrads_pos

    cdef RandomPool random_pool

    def __init__(self, nn, converter, options):
        """
        Initializes a new neural network initialized for training.
        :param options: provides
        :param left_context: defaut 1
        :param right_context: defaut 1
        :param hidden_size: default 20
        :param ngrams: size of ngrams to extract
        :param alpha: default 0.5
        """
        super(SentimentTrainer, self).__init__(nn, converter, options)

        self.alpha = options.get('alpha', 0.5)

        # cumulative AdaGrad
        self.input_adagrads_neg = np.zeros(self.nn.input_size, dtype=float)
        self.input_adagrads_pos = np.zeros(self.nn.input_size, dtype=float)

    @cython.boundscheck(False)
    cdef _train_pair_s(self, np.ndarray[INT_t,ndim=2] example, Gradients grads,
                       int size=1, int polarity=0):
        """
        Trains the network with a pair of positive/negative examples.
        The negative one is randomly generated.
	:param example: the positive example, i.e. a list of a list of token IDs
        :param grads: the computed gradients are accumulated here, except for
        inputs, which are updated immediately.
	:param size: size of ngram to generate for replacing window center
        :param polarity: 1 for positive, -1 for negative sentences, 0 for neutral..
        """
        
        # a token is a list of feature IDs.
        # token[0] is the list with the WordDictionary index of the word,
        cdef int left_context = len(self.pre_padding)
        cdef np.ndarray[INT_t,ndim=1] middle_token = example[left_context]
        cdef np.ndarray[INT_t,ndim=1] variant

        # ensure to generate a different word
        while True:
            variant = self.random_pool.next()
            if variant[0] != middle_token[0]:
                break

        cdef Network nn = self.nn
        cdef Parameters p = nn.p
        # FIXME: avoid allocation
        vars_pos = nn.variables()
        self.converter.lookup(example, vars_pos.input)
        nn.forward(vars_pos)

        vars_neg = nn.variables()
        cdef np.ndarray[INT_t,ndim=1] negative_token = np.array(variant, dtype=int)
        #print >> sys.stderr, 'pos', self.converter.extractors[0].sentence(example).encode('utf-8') # DEBUG
        #print >> sys.stderr, vars_pos.input[128:132] # DEBUG
        example[left_context] = negative_token
        self.converter.lookup(example, vars_neg.input)
        #print >> sys.stderr, 'neg', self.converter.extractors[0].sentence(example).encode('utf-8') # DEBUG
        #print >> sys.stderr, vars_neg.input[128:132] # DEBUG
        nn.forward(vars_neg)
        
        cdef float errorCW = max(0, 1 - vars_pos.output[0] + vars_neg.output[0])
        cdef float errorUS = max(0, 1 - polarity * vars_pos.output[1] + polarity * vars_neg.output[1])
        cdef float error = self.alpha * errorCW + (1 - self.alpha) * errorUS
        #print >> sys.stderr, 'error', errorCW, errorUS, error # DEBUG
        self.error += error
        self.avg_error.add(error) # moving average
        self.total_pairs += 1
        if error == 0: 
            self.skips += 1
            return error, variant
        
        # Compute the gradients

        # negative gradient for the positive example is +1, for the negative one is -1
        # (remember the network still has the values of the negative example) 

        # @see A.8 in Collobert et al. 2011.
        cdef np.ndarray[FLOAT_t,ndim=1] grads_pos_score = np.array([0.0, 0.0])
        cdef np.ndarray[FLOAT_t,ndim=1] grads_neg_score = np.array([0.0, 0.0])
        if (errorCW > 0):
            grads_pos_score[0] = 1.0
            grads_neg_score[0] = -1.0
        if (errorUS > 0):
            grads_pos_score[1] = 1.0
            grads_neg_score[1] = -1.0
        
        # grads.output_bias = grads_score
        # grads.output_weights = grads_score.T * hidden_values
        # grads_hidden = activationError(hidden_values) * grads_score.T.dot(output_weights)
        # grads.hidden_bias = grads_hidden
        # grads.hidden_weights = grads_hidden.T * input_values
        # grads.input = grads_hidden.dot(hidden_weights)

        # Output layer
        # CHECKME: summing they cancel each other:
        grads.output_bias = grads_pos_score + grads_neg_score
        # (2, hidden_size) = (2) x (hidden_size)
        grads.output_weights = np.outer(grads_pos_score, vars_pos.hidden) +\
                               np.outer(grads_neg_score, vars_neg.hidden)
        #print >> sys.stderr, 'gow', grads.output_weights[0,128:132] # DEBUG

        # Hidden layer
        # (hidden_size) = (2) * (2, hidden_size)
        grads.hidden_pos = hardtanhe(vars_pos.hidden) * grads_pos_score.dot(p.output_weights)
        grads.hidden_neg = hardtanhe(vars_neg.hidden) * grads_neg_score.dot(p.output_weights)

        # Input layer
        # (hidden_size) x (input_size) = (hidden_size, input_size)
        grads.hidden_weights = np.outer(grads.hidden_pos, vars_pos.input) +\
                               np.outer(grads.hidden_neg, vars_neg.input)

        #print >> sys.stderr, 'ghw', grads.hidden_weights[0,128:132] # DEBUG
        grads.hidden_bias = grads.hidden_pos + grads.hidden_neg

        return error, variant

    cdef _update(self, Gradients grads, float remaining,
                 np.ndarray[INT_t,ndim=2] example,
                 np.ndarray[INT_t,ndim=1] middle_token,
                 np.ndarray[INT_t,ndim=1] negative_token):
        """
        Update the weights along the gradients :param grads:
        """
        #cdef float LR_0 = max(0.001, self.learning_rate * remaining)
        # AdaGrad
        cdef float LR_0 = self.learning_rate
        cdef float LR_1 = max(0.001, self.learning_rate / self.nn.input_size * remaining)
        cdef float LR_2 = max(0.001, self.learning_rate / self.nn.hidden_size * remaining)

        cdef Network nn = self.nn
        cdef Parameters p = nn.p
        cdef int left_context = len(self.pre_padding)

        p.output_weights += LR_2 * grads.output_weights
        p.output_bias += LR_2 * grads.output_bias
        
        p.hidden_weights += LR_1 * grads.hidden_weights
        p.hidden_bias += LR_1 * grads.hidden_bias
        
        cdef np.ndarray[FLOAT_t,ndim=1] grads_input_neg, grads_input_pos

        # input gradients
	# (hidden_size) x (hidden_size, input_size) = (input_size)
        # grads.hidden_pos.dot(p.hidden_weights, grads.input)
        # grads.hidden_neg.dot(p.hidden_weights, grads.input_neg)
        # AdaGrad: cumulate the square of gradients in G for parameter p:
        # G += g^2
        # p -= LR * g / sqrt(G + eps)
        global adaEps
        self.input_adagrads_pos += np.power(grads.input, 2)
        self.input_adagrads_neg += np.power(grads.input_neg, 2)
        grads_input_pos = LR_0 * grads.input / np.sqrt(self.input_adagrads_pos + adaEps)
        grads_input_neg = LR_0 * grads.input_neg / np.sqrt(self.input_adagrads_neg + adaEps)
        #grads_input_pos = LR_0 * grads.input # DEBUG
        #grads_input_neg = LR_0 * grads.input_neg # DEBUG

        cdef int i, j, embeddings_size
        cdef np.ndarray[INT_t,ndim=1] token
        cdef np.ndarray[FLOAT_t,ndim=2] table
        cdef np.ndarray[FLOAT_t,ndim=1] deltas_neg, deltas_pos
             
        # this tracks where the deltas for the next table begins
        cdef int offset = 0
        for i, token in enumerate(example): 
            for j, e in enumerate(self.converter.extractors): # just one table
                table = e.table
                # i-th token in the window
                # j-th feature table (there is only one: j == 0)
                embeddings_size = table.shape[1]
                deltas_neg = grads_input_neg[offset: offset + embeddings_size]
                deltas_pos = grads_input_pos[offset: offset + embeddings_size]
                    
                if i == left_context:
                    # this is the middle position.
                    # apply negative and positive deltas to different tokens
                    table[negative_token[j]] += deltas_neg
                    table[middle_token[j]] += deltas_pos
                else:
                    # this is not the middle position. both deltas apply.
                    #print >> sys.stderr, token[j], deltas_neg, deltas_pos # DEBUG
                    table[token[j]] += deltas_neg + deltas_pos
                   
                offset += embeddings_size

    def train(self, Iterable sentences, list polarities, trie,
              int epochs, int report_freq):
        """
        Trains the sentiment language model on the given sentences.
        :param sentences: an iterable on a list of token features for each sentence
        :param iterations: number of train iterations
        :param polarities: the polarity of each sentence, +-1.
        :param trie: of ngrams
        """
        # FIXME: parallelize using ASGD.

        # generate 1000 random indices at a time to save time
        # (generating 1000 integers at once takes about ten times the time for a single one)
        # FIXME: nonsense to create random features besides ID
        feature_tables = [e.table for e in self.converter.extractors]
        self.random_pool = RandomPool([x.shape[0] for x in feature_tables])
        self.total_pairs = 0

        # how often to save model
        cdef int save_period = 1000 * 1000 # FIXME

        cdef float all_cases = float(sum([len(sen) for sen in sentences]) * epochs * self.ngram_size)

        cdef int epoch, num, pos
        cdef float remaining

        cdef int left_context = len(self.pre_padding)
        cdef int right_context = len(self.post_padding)
        cdef int window_size = left_context + 1 + right_context
        # FIXME: might use len(self.example) instead of window_size?
        cdef np.ndarray window = np.empty((window_size, 1), dtype=np.int)
        cdef np.ndarray token, neg_token
        cdef int size = 1

        grads = SentGradients(self.nn.input_size, self.nn.hidden_size, self.nn.output_size)

        for epoch in xrange(epochs):
            self.error = 0.0
            self.epoch_items = 0
            self.epoch_hits = 0
            self.skips = 0
            epoch_pairs = 0
            # update LR by fan-in
            # decrease linearly by remaining
            remaining = 1.0 - (self.total_pairs / all_cases)

            for num, sentence in enumerate(sentences):
                if polarities[num] == 0:
                    # skip neutral sentences
                    continue
                #print >> sys.stderr, self.converter.extractors[0].sentence(sentence).encode('utf-8') # DEBUG
                for pos in xrange(len(sentence)):
                    # for any word or ngram at sentence[pos:pos+size]
                    for size in itertrie(trie, sentence, pos):
                        # FIXME: avoid overlaps like 0-3, 1-2
                        # extract a window of tokens around the given position
                        token = self._extract_window(window, sentence, pos, size)
                        error, neg_token = self._train_pair_s(window, grads, size, polarities[num])
                        self.total_items += 1
                        if error > self.skipErr:
                            self.error += error
                            self._update(grads, remaining, window, token, neg_token)
                        else:
                            self.epoch_hits += 1
                        epoch_pairs += 1

                        if report_freq > 0 and \
                           (self.total_pairs and
                            self.total_pairs % report_freq == 0):
                            self._progress_report(epoch, self.total_pairs, num)
                            # periodically save language model
                            if save_period and self.total_pairs % save_period == 0:
                                self.saver(self)
            self._epoch_report(epoch + 1)
