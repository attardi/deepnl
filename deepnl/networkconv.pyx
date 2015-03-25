# -*- coding: utf-8 -*-
# distutils: language = c++

"""
A convolutional neural network for NLP tagging tasks like SRL.
It employs feature tables to store feature vectors for each token.
"""

import logging
from itertools import izip

from math import *
from network cimport *

# for decorations
cimport cython

cdef class ConvolutionalNetwork(Network):
    
    def __init__(self, feature_tables, target_dist_table, pred_dist_table, 
                 int word_window, int hidden1_size, int hidden2_size,
                 int output_size):
        """Creates a new convolutional neural network."""

        self.word_window = word_window
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        # sum the number of features in all tables 
        self.input_size = sum(table.shape[1] for table in feature_tables)
        # distance tables's input is treated differently
        self.input_size *= word_window

        # creates the weight matrices
        high = 2.38 / np.sqrt(input_size) # [Bottou-88]
        self.hidden_weights = np.random.uniform(-high, high, (hidden1_size, input_size))
        high = 2.38 / np.sqrt(hidden1_size) # [Bottou-88]
        self.hidden_bias = np.random.uniform(-high, high, hidden1_size)
        
        num_dist_features = word_window * target_dist_table.shape[1]
        self.target_dist_weights = np.random.uniform(-high, high, (num_dist_features, hidden1_size))
        num_dist_features = word_window * pred_dist_table.shape[1]
        self.pred_dist_weights = np.random.uniform(-high, high, (num_dist_features, hidden1_size))
        
        if hidden2_size > 0:
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
        
        self.feature_tables = feature_tables
        self.target_dist_table = target_dist_table
        self.pred_dist_table = pred_dist_table
        
    def description(self):
        """Returns a textual description of the network."""
        hidden2_size = 0 if self.hidden2_weights is None else self.hidden2_size
        table_dims = [str(t.shape[1]) for t in self.feature_tables]
        table_dims =  ', '.join(table_dims)
        
        dist_table_dims = '%d, %d' % (self.target_dist_table.shape[1], self.pred_dist_table.shape[1])
        
        desc = """
Word window size: %d
Feature table sizes: %s
Distance table sizes (target and predicate): %s
Input layer size: %d
Convolution layer size: %d 
Second hidden layer size: %d
Output size: %d
""" % (self.word_window_size, table_dims, dist_table_dims, self.input_size, self.hidden_size,
       hidden2_size, self.output_size)
        
        return desc
    
    
    def __init__(self, word_window, input_size, hidden1_size, hidden2_size,
                 output_size, hidden1_weights, hidden1_bias, target_dist_weights, 
                 pred_dist_weights, hidden2_weights, hidden2_bias, 
                 output_weights, output_bias):
        super(ConvolutionalNetwork, self).__init__(word_window, input_size, 
                                                   hidden1_size, output_size, 
                                                   hidden1_weights, hidden1_bias, 
                                                   output_weights, output_bias)
        self.half_window = word_window / 2
        self.features_per_token = self.input_size / word_window
        
        self.transitions = None
        self.target_dist_lookup = None
        self.pred_dist_lookup = None
        self.target_dist_weights = target_dist_weights
        self.pred_dist_weights = pred_dist_weights
        
        self.hidden2_size = hidden2_size
        self.hidden2_weights = hidden2_weights
        self.hidden2_bias = hidden2_bias
        
    def save(self, filename):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes, padding and 
        distance tables, but not other feature tables.
        """
        np.savez(filename, hidden_weights=self.hidden_weights,
                 target_dist_table=self.target_dist_table,
                 pred_dist_table=self.pred_dist_table,
                 target_dist_weights=self.target_dist_weights,
                 pred_dist_weights=self.pred_dist_weights,
                 output_weights=self.output_weights,
                 transitions=self.transitions,
                 hidden_bias=self.hidden_bias, output_bias=self.output_bias,
                 word_window_size=self.word_window_size, 
                 input_size=self.input_size, hidden_size=self.hidden_size,
                 output_size=self.output_size, hidden2_size=self.hidden2_size,
                 hidden2_weights=self.hidden2_weights, hidden2_bias=self.hidden2_bias,
                 padding_left=self.padding_left, padding_right=self.padding_right)

    @classmethod
    def load(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes, padding and 
        distance tables, but not other feature tables.
        """
        data = np.load(filename)
        
        # cython classes don't have the __dict__ attribute
        # so we can't do an elegant self.__dict__.update(data)
        hidden_weights = data['hidden_weights']
        hidden_bias = data['hidden_bias']
        hidden2_weights = data['hidden2_weights']
        
        # numpy stores None as an array containing None and with empty shape
        if hidden2_weights.shape == (): hidden2_weights = None
        
        hidden2_bias = data['hidden2_bias']
        output_weights = data['output_weights']
        output_bias = data['output_bias']
        
        word_window = data['word_window_size']
        input_size = data['input_size']
        hidden_size = data['hidden_size']
        hidden2_size = data['hidden2_size']
        output_size = data['output_size']
        
        nn = ConvolutionalNetwork(word_window, input_size, hidden_size, hidden2_size, 
                                  output_size, hidden_weights, hidden_bias, 
                                  data['target_dist_weights'], data['pred_dist_weights'],
                                  hidden2_weights, hidden2_bias, 
                                  output_weights, output_bias)
        
        nn.target_dist_table = data['target_dist_table']
        nn.pred_dist_table = data['pred_dist_table']
        transitions = data['transitions']
        nn.transitions = transitions if transitions.shape != () else None
        nn.padding_left = data['padding_left']
        nn.padding_right = data['padding_right']
        nn.pre_padding = np.array((nn.word_window_size / 2) * [nn.padding_left])
        nn.pos_padding = np.array((nn.word_window_size / 2) * [nn.padding_right])
        
        return nn
    
    def train(self, list sentences, list predicates, list tags,  
              int epochs, int epochs_between_reports=0,
              float desired_accuracy=0, list arguments=None):
        """
        Trains the convolutional network. Refer to the basic Network
        train method for detailed explanation.
        
        :param predicates: a list of 1-dim numpy array
            indicating the indices of predicates in each sentence.
        :param arguments: (only for argument classifying) a list of 2-dim
            numpy arrays indicating the start and end of each argument. 
        """
        self.only_classify = arguments is not None
        
        logger = logging.getLogger("Logger")
        logger.info("Training for up to %d epochs" % epochs)
        top_accuracy = 0
        last_accuracy = 0
        min_error = np.Infinity 
        last_error = np.Infinity
        
        cdef int i
        for i in xrange(epochs):
            self._train_epoch(sentences, predicates, tags, arguments)

            self.error = self.error / self.train_items
            self.accuracy = float(self.train_hits) / self.total_items
            
            if self.accuracy > top_accuracy:
                top_accuracy = self.accuracy
            
            if self.error < min_error:
                min_error = self.error
            
            if (epochs_between_reports > 0 and i % epochs_between_reports == 0) \
                or self.accuracy >= desired_accuracy > 0 \
                or (self.accuracy < last_accuracy and self.error > last_error):
                
                self._print_epoch_report(i + 1)

                if self.accuracy >= desired_accuracy > 0:
                    break
                
                if self.accuracy < last_accuracy and self.error > last_error:
                    # accuracy is falling, the network is probably diverging
                    break
            
            last_accuracy = self.accuracy
            last_error = self.error
        
        self.error = 0
        self.train_hits = 0
        self.total_items = 0

    def _train_epoch(self, sentences, predicates, tags, arguments):
        """Trains for one epoch with all examples."""
        self.train_hits = 0
        self.error = 0
        self.total_items = 0
        self.skips = 0
        
        # shuffle data
        # get the random number generator state in order to shuffle
        # sentences and their tags in the same order
        random_state = np.random.get_state()
        np.random.shuffle(sentences)
        np.random.set_state(random_state)
        np.random.shuffle(predicates)
        np.random.set_state(random_state)
        np.random.shuffle(tags)
        if arguments is not None:
            np.random.set_state(random_state)
            np.random.shuffle(arguments)
            i_args = iter(arguments)
        else:
            sent_args = None
        
        # keep last 2% for validation
        validation = int(len(sentences) * 0.98)

        for sent, sent_preds, sent_tags in izip(sentences, predicates, tags):
            if arguments is not None:
                sent_args = i_args.next()
            self._tag_sentence_conv(sent, sent_preds, sent_tags, sent_args)

            self.train_items += 1
    
    def tag_sentence(self, np.ndarray sentence, np.ndarray predicates, 
                     list arguments=None, bool logprob=False,
                     bool allow_repeats=True):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        
        :param sentence: a 2-dim numpy array, where each item encodes a token.
        :param predicates: a 1-dim numpy array, indicating the position
            of the predicates in the sentence
        :param logprob: a boolean indicating whether to return the 
            log-probability for each answer or not.
        :param allow_repeats: a boolean indicating whether to allow repeated
            argument classes (only for separate argument classification).
        """
        self.only_classify = arguments is not None
        return self._tag_sentence_conv(sentence, tags=None,
                                  predicates=predicates, arguments=arguments, 
                                  logprob=logprob, allow_repeats=allow_repeats)
    
    cdef np.ndarray argument_distances(self, positions, argument):
        """
        Calculates the distance from each token in the sentence to the argument.
        """
        distances = positions.copy()
        
        # the ones before the argument
        lo = np.less(positions, argument[0])
        distances[lo] -= argument[0]
        
        # the ones after the argument
        hi = np.greater(positions, argument[1])
        distances[hi] -= argument[1]
        
        # the ones inside the argument
        distances[np.logical_not(hi | lo)] = 0
        
        return distances
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[FLOAT_t,ndim=2] _tag_sentence_conv(self, np.ndarray[INT_t,ndim=2] sentence,
                      list tags=None,
                      np.ndarray predicates=None, list arguments=None, 
                      bool logprob=False, bool allow_repeats=True):
        """
        Runs the network for every predicate in the sentence.
        Refer to the Network class for more information.
        
        :param tags: this is a list rather than a numpy array because in
            argument classification, each predicate may have a differente number
            of arguments.
        :param predicates: a numpy array with the indices of the predicates in the sentence.
        """
        cdef np.ndarray[INT_t,ndim=1] answer = np.empty(len(sentence), dtype=np.int)
        cdef np.ndarray[FLOAT_t,ndim=2] convolution_lookup
        
        cdef bool train = tags is not None
        if train:
            # this table will store the values of the neurons for each input token
            # they will be needed during weight adjustments
            self.input_sent_values = np.empty((len(sentence), self.input_size))
            
        # store the convolution values to save time
        convolution_lookup = self._convolution_lookup(sentence, train)
        cdef np.ndarray[FLOAT_t,ndim=2] target_dist_features, pred_dist_features
        
        # store the values found by each convolution neuron here and then find the max
        cdef np.ndarray[FLOAT_t,ndim=2] convolution_values
        
        # store the a priori scores for each token
        cdef np.ndarray[FLOAT_t,ndim=2] scores
        cdef np.ndarray[FLOAT_t,ndim=1] token_scores
        
        if self.target_dist_lookup is None: self._create_target_lookup()
        if self.pred_dist_lookup is None: self._create_pred_lookup()
        if train: iter_tags = iter(tags)
        if self.only_classify:
            iter_args = iter(arguments)
        else:
            pred_arguments = None
        
        for predicate in predicates:
            
            if self.only_classify: pred_arguments = iter_args.next()
            
            self.num_targets = len(sentence) if arguments is None else len(pred_arguments)
            scores = np.empty((self.num_targets, self.output_size))
            
            if train: 
                pred_tags = iter_tags.next()
                # layer 2: results after applying hidden weights, before tanh
                # hidden sent values: results after tanh
                self.layer2_sent_values = np.empty((self.num_targets, self.hidden_size))
                self.hidden_sent_values = np.empty((self.num_targets, self.hidden_size))
                self.max_indices = np.empty((self.num_targets, self.hidden_size), np.int)
                if self.hidden2_weights is not None:
                    # layer 3: analogous to layer 2
                    self.layer3_sent_values = np.empty((self.num_targets, self.hidden2_size))
                    self.hidden2_sent_values = np.empty((self.num_targets, self.hidden2_size))
        
            # predicate distances are the same across all targets
            pred_dist_indices = np.arange(len(sentence)) - predicate
            pred_dist_features = self.pred_dist_lookup.take(pred_dist_indices + self.pred_dist_offset,
                                                            0, mode='clip')
            pred_dist_values = pred_dist_features.dot(self.pred_dist_weights)
            
            # add the weighted distance features to each token 
            for target in range(self.num_targets):
                
                # distance features for each window
                # if we are classifying all tokens, pick the distance to the target
                # if we are classifying arguments, pick the distance to the closest boundary 
                # of the argument (beginning or end)
                if arguments is None:
                    target_dist_indices = np.arange(len(sentence)) - target
                else:
                    argument = pred_arguments[target]
                    target_dist_indices = self.argument_distances(np.arange(len(sentence)), argument)
                
                target_dist_features = self.target_dist_lookup.take(target_dist_indices + self.target_dist_offset,
                                                                    0, mode='clip')

                convolution_values = target_dist_features.dot(self.target_dist_weights) \
                                     + pred_dist_values + convolution_lookup
                
                # now, find the maximum values
                if train:
                    self.max_indices[target] = convolution_values.argmax(0)
                
                # apply the bias and proceed to the next layer
                self.hidden_values = convolution_values.max(0) + self.hidden_bias
                if train:
                    self.hidden_sent_values[target] = self.hidden_values
                
                if self.hidden2_weights is not None:
                    self.hidden2_values = self.hidden2_weights.dot(self.hidden_values) + self.hidden2_bias
                    if train:
                        self.layer3_sent_values[target] = self.hidden2_values

                    self.hidden2_values = hardtanh(self.hidden2_values)
                    if train:
                        self.hidden2_sent_values[target] = self.hidden2_values
                else:
                    # apply non-linearity here
                    if train:
                        self.layer2_sent_values[target] = self.hidden_values
                    self.hidden2_values = hardtanh(self.hidden_values)
                    if train:
                        self.hidden_sent_values[target] = self.hidden2_values
                
                token_scores = self.output_weights.dot(self.hidden2_values)
                token_scores += self.output_bias
                scores[target] = token_scores
            
            pred_answer = self._viterbi_conv(scores, allow_repeats)
            
            if train:
                self._evaluate(pred_answer, pred_tags)
                if self._calculate_gradients(pred_tags, scores):
                    self._backpropagate_conv(sentence)
                    self._calculate_input_deltas(sentence, predicate, pred_arguments)
                    self._adjust_weights(predicate, pred_arguments)
                    self._adjust_features(sentence, predicate)
                
            if logprob:
                if self.only_classify:
                    raise NotImplementedError('Confidence measure not implemented for argument classifying')
                
                all_scores = self._calculate_delta(scores)
                last_token = len(sentence) - 1
                logadd = logsumexp(all_scores[last_token])
                confidence = self.answer_score - logadd
                pred_answer = (pred_answer, confidence)
            
            answer.append(pred_answer)
            
        return answer
    
    def _evaluate(self, answer, tags):
        """
        Evaluates the network performance, updating its hits count.
        """
        for net_tag, gold_tag in zip(answer, tags):
            if net_tag == gold_tag:
                self.train_hits += 1
        self.total_items += len(tags)
    
    def _calculate_gradients(self, tags, scores):
        """Delegates the call to the appropriate function."""
        if self.only_classify:
            return self._calculate_gradients_classify(tags, scores)
        else:
            return self._calculate_gradients_sll(tags, scores)
    
    def _calculate_gradients_classify(self, tags, scores):
        """
        Calculates the output deltas for each target in a network that only 
        classifies predelimited arguments.
        The aim is to minimize the cost, for each argument:
        logadd(score for all possible tags) - score(correct tag)
        
        :returns: whether a correction is necessary or not.
        """
        self.net_gradients = np.zeros_like(scores, np.float)
        correction = False
        
        for i, tag_scores in enumerate(scores):
            tag = tags[i]
            
            logadd = logsumexp(tag_scores)
            
            # update the total error 
            error = logadd - tag_scores[tag]
            self.error += error
            
            # like the non-convolutional network, don't adjust weights if the error
            # is too low. An error of 0.01 means a log-prob of -0.01 for the right
            # tag, i.e., more than 99% probability
            if error <= 0.01:
                self.skips += 1
                continue
            
            correction = True
            # negative gradients: will be added instead of subtracted
            self.net_gradients[i] = - np.exp(tag_scores - logadd)
            self.net_gradients[i, tag] += 1
    
        return correction

    cdef _backpropagate_conv(self, sentence):
        """Backpropagates the error gradient."""

        # this function only determines the gradients at each layer, without 
        # adjusting weights. This is done because the input features must 
        # be adjusted with the first weight matrix unchanged. 
        
        # derivative with respect to the non-linearity layer (tanh)
        dCd_tanh = self.net_gradients.dot(self.output_weights)
        
        if self.hidden2_weights is not None:
            # derivative with respect to the second hidden layer
            # the derivative of tanh(x) is 1 - tanh^2(x)
            dCd_hidden2 = dCd_tanh * hardtanhd(self.layer3_sent_values) 
            self.hidden2_gradients = dCd_hidden2
                        
            self.hidden_gradients = self.hidden2_gradients.dot(self.hidden2_weights)
        else:
            # the non-linearity appears right after the convolution max
            self.hidden_gradients = dCd_tanh * hardtanhd(self.layer2_sent_values)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _adjust_weights(self, predicate, arguments=None):
        """Adjusts the network weights after gradients have been calculated."""
        cdef int i
        cdef np.ndarray[FLOAT_t,ndim=1] gradients_t
        cdef np.ndarray[FLOAT_t,ndim=2] last_values, deltas, grad_matrix, input_values

        last_values = self.hidden2_sent_values if self.hidden2_weights is not None else self.hidden_sent_values
        
        deltas = self.net_gradients.T.dot(last_values) * self.learning_rate
        self.output_weights += deltas
        self.output_bias += self.net_gradients.sum(0) * self.learning_rate
        
        if self.hidden2_weights is not None:
            deltas = self.hidden2_gradients.T.dot(self.hidden_sent_values) * self.learning_rate
            self.hidden2_weights += deltas
            self.hidden2_bias += self.hidden2_gradients.sum(0) * self.learning_rate
        
        # now adjust weights from input to convolution. these will be trickier.
        # we need to know which input value to use in the delta formula
        
        # I tried vectorizing this loop but it got a bit slower, probably because
        # of the overhead in building matrices/tensors with the max indices
        for i, neuron_maxes in enumerate(self.max_indices):
            # i indicates the i-th target 
            gradients_t = self.hidden_gradients[i] * self.learning_rate
            
            # table containing in each line the input values selected for each convolution neuron
            input_values = self.input_sent_values.take(neuron_maxes, 0)
            
            # stack the gradients to multiply all weights for a neuron
            grad_matrix = np.tile(gradients_t, [self.input_size, 1]).T
            self.hidden_weights += grad_matrix * input_values
            
            # target distance weights
            # get the relative distance from each max token to its target
            if arguments is None:
                target_dists = neuron_maxes - i
            else:
                argument = arguments[i]
                target_dists = self.argument_distances(neuron_maxes, argument)
            
            dist_features = self.target_dist_lookup.take(target_dists + self.target_dist_offset, 
                                                         0, mode='clip')
            grad_matrix = np.tile(gradients_t, [self.target_dist_weights.shape[0], 1]).T
            self.target_dist_weights += (grad_matrix * dist_features).T
            
            # predicate distance weights
            # get the distance from each max token to its predicate
            pred_dists = neuron_maxes - predicate
            dist_features = self.pred_dist_lookup.take(pred_dists + self.pred_dist_offset,
                                                       0, mode='clip')
            # try to recycle the grad_matrix if sizes match
            if self.target_dist_weights.shape[0] != self.pred_dist_weights.shape[0]: 
                grad_matrix = np.tile(gradients_t, [self.pred_dist_weights.shape[0], 1]).T
            
            self.pred_dist_weights += (grad_matrix * dist_features).T
        
        self.hidden_bias += self.hidden_gradients.sum(0) * self.learning_rate

        # Adjusts the transition scores table with the calculated gradients.
        if not self.only_classify and self.transitions is not None:
            self.transitions += self.trans_gradients * self.learning_rate_trans
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _calculate_input_deltas(self, sentence, predicate, arguments=None):
        """Calculates the input deltas to be applied in the feature tables."""
        cdef np.ndarray[FLOAT_t,ndim=2] grad_matrix, hidden_gradients
        
        # this gradient matrix has a whole window in each line
        self.input_deltas = np.zeros((len(sentence), self.input_size))
        self.target_dist_deltas = np.zeros_like(self.target_dist_lookup, np.float)
        self.pred_dist_deltas = np.zeros_like(self.pred_dist_lookup, np.float)
        
        # avoid multiplying by the learning rate multiple times
        hidden_gradients = self.hidden_gradients * self.learning_rate_features
        
        for target in range(self.num_targets):
            # the token that yielded the maximum value in each neuron
            convolution_max = self.max_indices[target]
            
            if not self.only_classify:
                target_dists = convolution_max - target
            else:
                argument = arguments[target]
                target_dists = self.argument_distances(convolution_max, argument)
            
            target_dists = np.clip(target_dists + self.target_dist_offset, 0,
                                   self.target_dist_lookup.shape[0] - 1)
            pred_dists = convolution_max - predicate
            pred_dists = np.clip(pred_dists + self.pred_dist_offset, 0,
                                 self.pred_dist_lookup.shape[0] - 1)
            
            gradients = hidden_gradients[target]
            
            # sparse matrix with gradients to be applied over the input
            # line i has the gradients for the i-th token in the sentence
            grad_matrix = np.zeros((len(sentence), self.hidden_size)) 
            grad_matrix[convolution_max, np.arange(self.hidden_size)] = gradients
            
            self.input_deltas += grad_matrix.dot(self.hidden_weights) 
            
            # distance deltas
            grad_matrix = np.zeros((self.target_dist_lookup.shape[0], self.hidden_size))
            grad_matrix[target_dists, np.arange(self.hidden_size)] = gradients
            self.target_dist_deltas += grad_matrix.dot(self.target_dist_weights.T)
            
            grad_matrix = np.zeros((self.pred_dist_lookup.shape[0], self.hidden_size))
            grad_matrix[pred_dists, np.arange(self.hidden_size)] = gradients
            self.pred_dist_deltas += grad_matrix.dot(self.pred_dist_weights.T)
            
        
    def _adjust_features(self, sentence, predicate):
        """Adjusts the features in all feature tables."""
        # compute each token in the window separately and
        # separate the feature deltas into tables
        start_from = 0
        dist_target_from = 0
        dist_pred_from = 0
        
        # number of times that the minimum and maximum distances are repeated
        # in the lookup distance tables
        pre_dist = self.word_window_size
        pos_dist = 1
        if self.word_window_size > 1:
            padded_sentence = np.concatenate((self.pre_padding,
                                              sentence,
                                              self.pos_padding))
        else:
            padded_sentence = sentence
        
        for i in range(self.word_window_size):
            
            for j, table in enumerate(self.feature_tables):
                # this is the column for the i-th position in the window
                # regarding features from the j-th table
                table_deltas = self.input_deltas[:, start_from:start_from + table.shape[1]]
                start_from += table.shape[1]
                
                for token, deltas in zip(padded_sentence[i:], table_deltas):
                    table[token[j]] += deltas
            
            dist_deltas = self.target_dist_deltas[:, dist_target_from : dist_target_from + self.target_dist_table.shape[1] ]
            pre_deltas = dist_deltas.take(np.arange(pre_dist), 0).sum(0)
            pos_deltas = dist_deltas.take(np.arange(-pos_dist, 0), 0).sum(0)
            self.target_dist_table[1:-1, :] += dist_deltas[pre_dist : -pos_dist]
            self.target_dist_table[0] += pre_deltas
            self.target_dist_table[-1] += pos_deltas 
            dist_target_from += self.target_dist_table.shape[1]
            
            dist_deltas = self.pred_dist_deltas[:, dist_pred_from : dist_pred_from + self.pred_dist_table.shape[1] ]
            pre_deltas = dist_deltas.take(np.arange(pre_dist), 0).sum(0)
            pos_deltas = dist_deltas.take(np.arange(-pos_dist, 0), 0).sum(0)
            self.pred_dist_table[1:-1, :] += dist_deltas[pre_dist : -pos_dist]
            self.pred_dist_table[0] += pre_deltas
            self.pred_dist_table[-1] += pos_deltas
            
            pre_dist -= 1
            pos_dist += 1
            dist_pred_from += self.pred_dist_table.shape[1]
            
        
        self._create_target_lookup()
        self._create_pred_lookup()
    
    @cython.boundscheck(False)
    cdef np.ndarray[INT_t,ndim=1] _viterbi_conv(self, np.ndarray[FLOAT_t,ndim=2] scores, bool allow_repeats=True):
        """
        Performs a Viterbi search over the scores for each tag using
        the transitions matrix.
        :return: the most likely sequence of tokens.
            If no matrix was supplied, return the tags with the highest scores individually.
        """
        if self.transitions is None:
            best_scores = scores.argmax(1)
            
            if allow_repeats:
                return best_scores
            
            # we must find the combination of tags that maximizes the probabilities
            logadd = logsumexp(scores, 1)
            logprobs = (scores.T - logadd).T
            counts = np.bincount(best_scores)
            
            while counts.max() != 1:
                # find the tag with the most conflicting args
                conflicting_tag = counts.argmax()
                
                # arguments with that tag as current maximum
                args = np.where(best_scores == conflicting_tag)[0]
                
                # get the logprobs for those args having this tag
                conflicting_probs = logprobs[args, conflicting_tag]
                
                # find the argument with the highest probability for that tag
                highest_prob_arg = args[conflicting_probs.argmax()] 
                
                # set the score for other arguments in that tag to a low value
                other_args = args[args != highest_prob_arg]
                scores[other_args, conflicting_tag] = -1000
                
                # and find the new maxes, without recalculating probabilities
                best_scores = scores.argmax(1)
                counts = np.bincount(best_scores)
            
            return best_scores
        
        path_scores = np.empty_like(scores)
        path_backtrack = np.empty_like(scores, np.int)
        
        # now the actual Viterbi algorithm
        # first, get the scores for each tag at token 0
        # the last row of the transitions table has the scores for the first tag
        path_scores[0] = scores[0] + self.transitions[-1]
        
        for i, token in enumerate(scores[1:], 1):
            
            # each line contains the score until each tag t plus the transition to each other tag t'
            prev_score_and_trans = (path_scores[i - 1] + self.transitions[:-1].T).T
            
            # find the previous tag that yielded the max score
            path_backtrack[i] = prev_score_and_trans.argmax(0)
            path_scores[i] = prev_score_and_trans[path_backtrack[i], 
                                                  np.arange(self.output_size)] + scores[i]
            
        # now find the maximum score for the last token and follow the backtrack
        answer = np.empty(len(scores), dtype=np.int)
        answer[-1] = path_scores[-1].argmax()
        self.answer_score = path_scores[-1][answer[-1]]
        previous_tag = path_backtrack[-1][answer[-1]]
        
        for i in range(len(scores) - 2, 0, -1):
            answer[i] = previous_tag
            previous_tag = path_backtrack[i][previous_tag]
        
        answer[0] = previous_tag
        return answer
    
    def _create_target_lookup(self):
        """
        Creates a lookup table with the window value for each different distance
        to the target token. 
        """
        # consider padding. if the table has 10 entries, with a word window of 3,
        # we would have to consider up to the distance of 11, because of the padding.
        num_distances = self.target_dist_table.shape[0] + self.word_window_size - 1
        self.target_dist_lookup = np.empty((num_distances, 
                                            self.word_window_size * self.target_dist_table.shape[1]))
        self.target_dist_offset = num_distances / 2
        window_from = 0
        window_to = self.target_dist_table.shape[1] 
        for i in range(self.word_window_size):
            # each token in the window will is shifted in relation to the middle one
            shift = i - self.half_window
            
            # discount half window size because of the extra distances we added for padding
            inds = np.arange(shift, num_distances + shift) - self.half_window
            inds = np.clip(inds, 0, self.target_dist_table.shape[0] - 1)
            self.target_dist_lookup[:,window_from : window_to] = self.target_dist_table[inds,]
            
            window_from = window_to
            window_to += self.target_dist_table.shape[1] 
    
    def _create_pred_lookup(self):
        """
        Creates a lookup table with the window value for each different distance
        to the predicate token. 
        """
        # consider padding. if the table has 10 entries, with a word window of 3,
        # we would have to consider up to the distance of 11, because of the padding.
        num_distances = self.pred_dist_table.shape[0] + self.word_window_size - 1
        self.pred_dist_lookup = np.empty((num_distances, 
                                          self.word_window_size * self.pred_dist_table.shape[1]))
        self.pred_dist_offset = num_distances / 2
        window_from = 0
        window_to = self.pred_dist_table.shape[1] 
        for i in range(self.word_window_size):
            # each token in the window will is shifted in relation to the middle one
            shift = i - self.half_window
            
            # discount half window size because of the extra distances we added for padding
            inds = np.arange(shift, num_distances + shift) - self.half_window
            inds = np.clip(inds, 0, self.pred_dist_table.shape[0] - 1)
            self.pred_dist_lookup[:,window_from : window_to] = self.pred_dist_table[inds,]
            
            window_from = window_to
            window_to += self.pred_dist_table.shape[1] 
    
    def _convolution_lookup(self, sentence, train):
        """
        Creates a lookup table storing the values found by each
        convolutional neuron before summing distance features.
        The table has the format len(sent) x len(convol layer)
        Biases are not included.
        """
        cdef np.ndarray padded_sentence
        
        # add padding to the sentence
        if self.word_window_size > 1:
            padded_sentence = np.concatenate((self.pre_padding,
                                             sentence,
                                             self.pos_padding))
        else:
            padded_sentence = sentence
        
        cdef np.ndarray[FLOAT_t,ndim=2] lookup = np.empty((len(sentence), self.hidden_size))
        
        # first window
        cdef np.ndarray window = padded_sentence[:self.word_window_size]
        cdef np.ndarray input_data = self.lookup(window)
        
        lookup[0] = self.hidden_weights.dot(input_data)
        if train:
            # store the values of each input -- needed when adjusting features
            self.input_sent_values[0] = input_data
        
        cdef np.ndarray[FLOAT_t,ndim=1] new_data
        for i, element in enumerate(padded_sentence[self.word_window_size:], 1):
            new_data = np.concatenate([table[index] for 
                                       index, table in zip(element, self.feature_tables)])
            
            # slide the window to the next element
            input_data = np.concatenate((input_data[self.features_per_token:], 
                                         new_data))
            
            lookup[i] = self.hidden_weights.dot(input_data)
            if train:
                self.input_sent_values[i] = input_data
        
        return lookup
