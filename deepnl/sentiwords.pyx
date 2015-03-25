# -*- coding: utf-8 -*-
# distutils: language = c++

"""
Train a sentient specific language model.
"""

import numpy as np

# for decorations
cimport cython

# local
from math cimport *
from words cimport *
from extractors cimport Iterable

# ----------------------------------------------------------------------

cdef class SentGradients(Gradients):

    cdef public np.ndarray pos_hidden
    cdef public np.ndarray neg_hidden
    
    def __init__(self, int input_size, int hidden_size, int output_size):
        super(SentGradients, self).__init__(input_size, hidden_size, output_size)
        # gradients for positive hidden variables
        self.pos_hidden = np.zeros(hidden_size, dtype=float)
        # gradients for nfative hidden variables
        self.neg_hidden = np.zeros(hidden_size, dtype=float)

    def clear(self):
        super(SentGradients, self).clear()
        self.pos_hidden.fill(0.0)
        self.neg_hidden.fill(0.0)

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

    cdef np.ndarray neg_hidden_adagrads
    cdef np.ndarray pos_hidden_adagrads

    def __init__(self, converter, float learning_rate,
                 int left_context, int right_context,
                 int hidden_size, int ngrams, float alpha):
        """
        Initializes a new neural network initialized for training.
        :param left_context: defaut 1
        :param right_context: defaut 1
        :param hidden_size: default 20
        :param alpha: default 0.5
        """
        super(SentimentTrainer, self).__init__(converter, learning_rate,
                                               left_context, right_context,
                                               hidden_size, 2)

        self.ngrams = ngrams
        self.alpha = alpha

        # cumulative AdaGrad
        self.neg_hidden_adagrads = np.zeros(hidden_size, dtype=float)
        self.pos_hidden_adagrads = np.zeros(hidden_size, dtype=float)

    @cython.boundscheck(False)
    cdef _train_pair_s(self, np.ndarray[INT_t,ndim=2] example, Gradients grads,
                       int size=1, int polarity=0):
        """
        Trains the network with a pair of positive/negative examples.
        The negative one is randomly generated.
	:param example: the positive example, i.e. a list of a list of token IDs
        :param grads: the computed gradients are accumulated here, except for 
	:param size: size of ngram to generate for replacing window center
        :param polarity: 1 for positive, -1 for negative sentences.
        """
        
        # a token is a list of feature IDs.
        # token[0] is the list with the WordDictionary index of the word,
        cdef np.ndarray[INT_t,ndim=1] middle_token = example[self.nn.left_context]
        cdef np.ndarray[INT_t,ndim=1] variant

        if size == 1:
	   # ensure to generate a different word
            while True:
                variant = self.random_pool.next()
                if variant[0] != middle_token[0]:
                    break
        else:
            raise Exception("ngram size not implemented")

        cdef Network nn = self.nn
        cdef np.ndarray[FLOAT_t,ndim=1] pos_input_values = nn.converter.lookup(example)
        cdef np.ndarray[FLOAT_t,ndim=1] pos_score = nn.run(pos_input_values)
        cdef np.ndarray[FLOAT_t,ndim=1] pos_hidden_values = nn.hidden_values

        cdef np.ndarray[INT_t,ndim=1] negative_token = np.array(variant, dtype=int)
        example[nn.left_context] = negative_token
        cdef np.ndarray[FLOAT_t,ndim=1] neg_input_values = nn.converter.lookup(example)
        cdef np.ndarray[FLOAT_t,ndim=1] neg_score = nn.run(neg_input_values)
        
        cdef float errorCW = max(0, 1 - pos_score[0] + neg_score[0])
        cdef float errorUS = max(0, 1 - polarity * pos_score[1] + polarity * neg_score[1])
        cdef float error = self.alpha * errorCW + (1 - self.alpha) * errorUS
        self.error += error
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
        # (2) x (hidden_size) = (2, hidden_size)
        grads.output_weights += np.outer(grads_pos_score, pos_hidden_values) + np.outer(grads_neg_score, nn.hidden_values)

        # Hidden layer
        # (2) x (2, hidden_size) = (hidden_size)
        grads.pos_hidden = hardtanhe(pos_hidden_values) * grads_pos_score.dot(nn.output_weights)
        grads.neg_hidden = hardtanhe(nn.hidden_values) * grads_neg_score.dot(nn.output_weights)

        # Input layer
        # (hidden_size) x (input_size) = (hidden_size, input_size)
        cdef np.ndarray[FLOAT_t,ndim=2] grads_pos_hidden_weights = np.outer(grads.pos_hidden, pos_input_values)
        cdef np.ndarray[FLOAT_t,ndim=2] grads_neg_hidden_weights = np.outer(grads.neg_hidden, neg_input_values)
        grads.hidden_weights = grads_pos_hidden_weights + grads_neg_hidden_weights
        grads.hidden_bias = grads.pos_hidden + grads.neg_hidden

        return error, variant

    cdef update(self, Gradients grads, float remaining,
                np.ndarray[INT_t,ndim=2] example,
                np.ndarray[INT_t,ndim=1] middle_token,
                np.ndarray[INT_t,ndim=1] negative_token):
        """
        Update the weights along the gradients :param grads:
        """
        cdef float LR_0 = max(0.001, self.learning_rate * remaining)
        cdef float LR_1 = max(0.001, self.learning_rate / self.nn.input_size * remaining)
        cdef float LR_2 = max(0.001, self.learning_rate / self.nn.hidden_size * remaining)

        cdef Network nn = self.nn

        nn.output_weights += LR_2 * grads.output_weights
        nn.output_bias += LR_2 * grads.output_bias
        
        nn.hidden_weights += LR_1 * grads.hidden_weights
        nn.hidden_bias += LR_1 * grads.hidden_bias
        
        # input gradients, using AdaGrad
        self.neg_hidden_adagrads += np.power(grads.neg_hidden, 2)
	# (hidden_size) x (hidden_size, input_size) = (input_size)
        grads_neg_input = (grads.neg_hidden / np.sqrt(self.neg_hidden_adagrads)).dot(nn.hidden_weights)

        self.pos_hidden_adagrads += np.power(grads.pos_hidden, 2)
        grads_pos_input = (grads.pos_hidden / np.sqrt(self.pos_hidden_adagrads)).dot(nn.hidden_weights)

        deltas_neg_input = LR_0 * grads_neg_input
        deltas_pos_input = LR_0 * grads_pos_input
        
        cdef int i, j
        cdef np.ndarray[INT_t,ndim=1] token
        cdef np.ndarray[FLOAT_t,ndim=2] table
             
        # this tracks where the deltas for the next table begins
        cdef int offset = 0
        for i, token in enumerate(example):
            for j, table in enumerate(nn.feature_tables): # just one table
                # i-th token in the window
                # j-th feature table (there is only one: j == 0)
                embeddings_size = table.shape[1]
                deltas_neg = deltas_neg_input[offset: offset + embeddings_size]
                deltas_pos = deltas_pos_input[offset: offset + embeddings_size]
                    
                if i == nn.left_context:
                    # this is the middle position.
                    # apply negative and positive deltas to different tokens
                    table[negative_token[j]] += deltas_neg
                    table[middle_token[j]] += deltas_pos
                else:
                    # this is not the middle position. both deltas apply.
                    table[token[j]] += deltas_neg + deltas_pos
                
                offset += embeddings_size

    def train(self, Iterable sentences, int epochs, int report_freq,
              list polarities, ngram_dict):
        """
        Trains the sentiment language model on the given sentences.
        :param sentences: an iterable on a list of token features for each sentence
        :param iterations: number of train iterations
        :param polarities: the polarity of each sentence, +-1.
        :param ngram_dict: the dictionary of the ngrams on the corpus
        """
        # generate 1000 random indices at a time to save time
        # (generating 1000 integers at once takes about ten times the time for a single one)
        self.random_pool = RandomPool([x.shape[0] for x in self.nn.feature_tables])
        self.total_pairs = 0

        # how often to save model
        cdef int save_period = 1000 * 1000

        cdef float all_cases = float(sum([len(sen) for sen in sentences]) * epochs * self.ngrams)

        cdef int epoch, epoch_examples, num, pos
        cdef float remaining

        grads = SentGradients(self.nn.input_size, self.nn.hidden_size, self.nn.output_size)

        for epoch in xrange(epochs):
            self.error = 0.0
            self.skips = 0
            epoch_examples = 0
            # update LR by fan-in
            # decrease linearly by remaining
            remaining = 1.0 - (self.total_pairs / all_cases)

            for num, sentence in enumerate(sentences):
                for pos in xrange(len(sentence)):
                
                    # ngram size changes periodically
                    if self.ngrams > 1 and  self.total_pairs:
                        if self.total_pairs % 5 == 0:
                            size = 2
                        elif self.total_pairs % 17 == 0:
                            size = 3
                        else:
                            size = 1
                    else:
                        size = 1

                    # extract a window of tokens around the given position
                    if size == 1:
                        token = sentence[pos]
                    else:
                        # get IDs of each token
                        ngrams = [sentence[i][0] for i in xrange(pos, pos + size)]
                        # lookup ngram IDs to obtain words
                        tokens = ngram_dict.get_words(ngrams)
                        token = np.array([ngram_dict[' '.join(tokens)]]) # one feature_table

                    window = self._extract_window(sentence, pos, token, size)

                    error, neg_token = self._train_pair_s(window, grads, size, polarities[num])
                    self.update(grads, remaining, window, token,
                                neg_token)
                    epoch_examples += 1

                    if report_freq > 0 and \
                       (self.total_pairs and
                        self.total_pairs % report_freq == 0):
                        self._progress_report(epoch_examples)
                        # periodically save language model
                        if save_period and self.total_pairs % save_period == 0:
                            self.saver(self)

    # @classmethod
    # def load_weights(cls, filename):
    #     # inherit from base class, but instantiate derived cls
    #     return LanguageModel.load_weights.__func__(cls, filename)
