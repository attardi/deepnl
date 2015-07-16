# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: profile=True

"""
Train a DL Neural Network.
"""

import numpy as np
import cPickle as pickle
import struct
import logging
from itertools import izip
import sys

# local
from network cimport *
from networkseq cimport *
from tagger cimport Tagger

# for decorations
cimport cython

# FIXME: defined also in trainerconv.pyx, networkseq.pyx
cdef float eps = 0.01          # minimum error worth an update

# ----------------------------------------------------------------------

cdef class MovingAverage(object):

    # cdef float mean
    # cdef float variance
    # cdef unsigned count

    def __init__(self):
        self.mean = 0.0
        self.variance = 0.0
        self.count = 0

    cdef add(self, float v):
        """Add value :param v: to the moving average."""
        self.count += 1
        self.mean += (2. / self.count) * (v - self.mean)
        cdef float this_variance = (v - self.mean) * (v - self.mean)
        self.variance += (2. / self.count) * (this_variance - self.variance)

# ----------------------------------------------------------------------

cdef class Trainer(object):
    """
    Abstract class for trainers.
    """

    # cdef readonly Network nn
    # cdef public Converter converter
    # cdef int left_context, right_context
    # cdef int ngram_size
    # cdef public float learning_rate
    # cdef public object saver
    # cdef int train_items, skips
    
    def __init__(self, Converter converter, float learning_rate,
                 int left_context, int right_context,
                 Network nn, int ngrams=1, bool verbose=False):
        """
        Creates a neural network initialized for training.
        """
        
        self.converter = converter
        self.learning_rate = learning_rate
        self.ngram_size = ngrams
        self.avg_error = MovingAverage()
        self.verbose = verbose
        self.saver = lambda x: None
        self.nn = nn        # dependency injection
        cdef np.ndarray padding_left = converter.get_padding_left()
        cdef np.ndarray padding_right = converter.get_padding_right()
        self.pre_padding = np.array(left_context * [padding_left])
        self.post_padding = np.array(right_context * [padding_right])

    def train(self, list examples, list outcomes, 
              int epochs, int report_frequency=0,
              float desired_accuracy=0, threads=1):
        """
        Trains the network to tag examples.
        
        :param examples: training examples.
        :param outcomes: outcome for each example.
        :param epochs: number of training epochs
        :param report_frequency: number of epochs to wait between
            reports about the training performance. 0 means no reports.
        :param desired_accuracy: training stops if the desired accuracy
            is reached. Ignored if 0.
        """
        logger = logging.getLogger("Logger")
        logger.info("Training for up to %d epochs" % epochs)
        top_accuracy = 0
        last_accuracy = 0
        min_error = np.Infinity 
        last_error = np.Infinity
        
        np.seterr(all='raise')

        for i in xrange(epochs):
            self._train_epoch(examples, outcomes)
            
            # normalize error
            self.nn.error = self.nn.error / self.train_items if self.train_items else np.Infinity
            # save model
            if self.nn.error < min_error:
                min_error = self.nn.error
                self.saver(self)

            if self.accuracy > top_accuracy:
                top_accuracy = self.accuracy
            
            if (report_frequency > 0 and i % report_frequency == 0) \
                or self.accuracy >= desired_accuracy > 0 \
                or (self.accuracy < last_accuracy and self.nn.error > last_error):
                self._epoch_report(i + 1)
                if self.accuracy >= desired_accuracy > 0:
                        #or (self.nn.error > last_error and self.accuracy < last_accuracy): # early stop DEBUG
                    break
                
            last_accuracy = self.accuracy
            last_error = self.nn.error

    def _train_epoch(self, list sentences, list labels):
        """
        Trains for one epoch with all examples.
        :param sentences: a list of 2-dim numpy arrays, where each item 
            encodes a sentence. Each item in a sentence has the 
            indices to its features.
        :param labels: a list of id of labels of each corresponding sentence.
        """

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
        np.random.shuffle(labels)
        
        # keep last 2% for validation
        validation = int(len(sentences) * 0.98)

        nn = self.nn
        vars = nn.variables()        # allocate variables
        ada = nn.gradients()
        cdef int i = 0
        for sent, label in izip(sentences, labels):
            self.converter.lookup(sent, vars.input)
            nn.run(vars)
            grads = nn.gradients(len(sent)) # allocate gradients
            loss = nn.backpropagate(label, vars, grads)
            if loss > eps:
                self.update(grads, self.learning_rate, sent, ada)
            else:
                self.skips += 1
            self.train_items += 1
            # progress report
            i += 1
            if self.verbose:
                if i%1000 == 0:
                    sys.stderr.write('+')
                    sys.stderr.flush()
                elif i%100 == 0:
                    sys.stderr.write('.')
                    sys.stderr.flush()
            if i == validation:
                break
        if self.verbose:
            sys.stderr.write('\n')

        self.accuracy = self._validate(sentences, labels, validation)

    @cython.boundscheck(False)
    cdef float _validate(self, list sentences, labels, int idx):
        """Perform validation on held out data and estimate accuracy
        :param idx: index of first sentence in validation set.
        """
        cdef int tokens = 0
        cdef int hits = 0

        cdef int i
        cdef np.ndarray[INT_t,ndim=2] sent
        cdef np.ndarray[FLOAT_t,ndim=2] scores

        for i in xrange(idx, len(sentences)):
            sent = sentences[i]
            label = labels[i]
            scores = self.nn.run(sent)
            if np.argmax(scores) == label:
                hits += 1
            tokens += 1
        if tokens:
            return float(hits) / tokens
        else:
            return 0.0

    def _epoch_report(self, int num):
        """
        Reports the status of the network in the given training epoch,
        including error and accuracy.
        """
        logger = logging.getLogger("Logger")
        msg = ""
        if self.skips:
            msg = "   %d corrections skipped" % self.skips
        logger.info("%d epochs   Error: %f   Accuracy: %f%s" \
                        % (num, self.nn.error, self.accuracy, msg))

    cpdef update(self, Gradients grads, float learning_rate,
                 np.ndarray[INT_t,ndim=2] sentence, Gradients ada=None):
        """
        Adjust the weights.
        :param ada: cumulative square gradients for performing AdaGrad.
        """

        # update network weights.
        self.nn.update(grads, learning_rate, ada)

        # adjust input weights
        cdef int window_size = len(self.left_context) + 1 + len(self.right_context)
        padded_sentence = np.concatenate((self.pre_padding,
                                          sentence,
                                          self.post_padding))
        grads.input *= learning_rate

        cdef int i
        for i in xrange(len(sentence)):
            window = padded_sentence[i: i+window_size]
            self.converter.update(window, grads.input[i])

    def save(self, file):
        np.save(file, (self.left_context, self.right_context, self.ngram_size))
        self.nn.save(file)
        self.converter.save(file)

    @classmethod
    def load(cls, file):
        """
        Resume training from previous dump.
        """
        trainer = Trainer.__new__(cls)
        trainer.left_context, trainer.right_context, trainer.ngram_size = np.load(file)
        trainer.nn = Network.load(file)
        trainer.converter = Converter.load(file)
        return trainer

# ----------------------------------------------------------------------

cdef class TaggerTrainer(Trainer):
    """
    A trainer for sequence taggers.
    """

    def __init__(self, Converter converter, float learning_rate,
                 int left_context, int right_context,
                 int hidden_size, tag_dict, verbose=False):
        # sum the number of features in all tables 
        input_size = converter.size()
        window_size = left_context + 1 + right_context
        input_size *= window_size
        output_size = len(tag_dict)
        nn = SequenceNetwork(input_size, hidden_size, output_size)

        self.tagger = Tagger(converter, tag_dict, left_context, right_context,
                             nn)
        super(TaggerTrainer, self).__init__(converter, learning_rate,
                                            left_context, right_context,
                                            nn, verbose=verbose)

    def _train_epoch(self, list sentences, list tags):
        """
        Trains for one epoch with all examples.
        :param sentences: a list of 2-dim numpy arrays, where each item 
            encodes a sentence. Each item in a sentence has the 
            indices to its features.
        :param tags: a list of 1-dim numpy arrays, where each item has
            the tags of the corresponding sentence.
        """
        # FIXHIM: should be a parametric type
        cdef SequenceNetwork nn = <SequenceNetwork>self.tagger.nn     # same as self.nn
        nn.error = 0
        self.skips = 0
        self.train_items = 0
        
        # shuffle data
        # get the random number generator state in order to shuffle
        # sentences and their tags in the same order
        random_state = np.random.get_state()
        np.random.shuffle(sentences)
        np.random.set_state(random_state)
        np.random.shuffle(tags)
        
        cdef SeqGradients grads
        # AdaGrad. Since sentence length varies, we accumulate into a single
        # item all squared input gradients
        cdef SeqGradients ada = nn.gradients(1)

        # keep last 2% for validation
        cdef int validation = int((len(sentences) - 1) * 0.98) + 1 # at least 1

        cdef int i = 0
        for sent, sent_tags in izip(sentences, tags):
            scores = self.tagger._tag_sequence(sent, True)
            grads = nn.gradients(len(sent))
            if nn._calculate_gradients_sll(sent_tags, grads, scores):
                nn._backpropagate(grads)
                self.update(grads, self.learning_rate, sent, ada)
            else:
                self.skips += 1

            self.train_items += len(sent)

            # progress report
            i += 1
            if self.verbose:
                if i%1000 == 0:
                    sys.stderr.write('+')
                    sys.stderr.flush()
                elif i%100 == 0:
                    sys.stderr.write('.')
                    sys.stderr.flush()
            if i == validation:
                break
        if self.verbose:
            sys.stderr.write('\n')

        self.accuracy = self._validate(sentences, tags, validation)
        # DEBUG
        # print >> sys.stderr, 'hw', nn.hidden_weights[:4,:4]
        # print >> sys.stderr, 'ow', nn.output_weights[0:4,:4]

    @cython.boundscheck(False)
    cdef float _validate(self, list sentences, tags, int idx):
        """Perform validation on held out data and estimate accuracy
        :param idx: index of first sentence in validation set.
        """
        cdef int tokens = 0
        cdef int hits = 0

        cdef int i
        cdef np.ndarray[INT_t,ndim=2] sent
        cdef np.ndarray[INT_t,ndim=1] answer
        cdef np.ndarray[FLOAT_t,ndim=2] scores

        for i in xrange(idx, len(sentences)):
            sent = sentences[i]
            gold_tags = tags[i]
            scores = self.tagger._tag_sequence(sent)
            answer = self.tagger.nn._viterbi(scores)
            for pred_tag, gold_tag in izip(answer, gold_tags):
                if pred_tag == gold_tag:
                    hits += 1
                tokens += 1
        if tokens:
            return float(hits) / tokens
        else:
            return 0.0

    cpdef update(self, Gradients grads, float learning_rate,
                 np.ndarray[INT_t,ndim=2] sentence, Gradients ada=None):

        # update network weights and transition weights.
        (<SequenceNetwork>self.nn).update(grads, learning_rate, ada)
        #
        # Adjust the features indexed by the input window.
        #
        # the deltas that will be applied to the feature tables
        # they are in the same sequence as the network receives them, i.e.
        # [token1-table1][token1-table2][token2-table1][token2-table2] (...)
        # e.g. num features = 50 (embeddings) + 5 (caps) + 5 (suffix) = 60
        # input_size = num features * window (e.g. 60 * 5).

        cdef int window_size = len(self.pre_padding) + 1 + len(self.post_padding)
        cdef int i
        cdef int slen = len(sentence)

        # (len, input_size)
        cdef np.ndarray[FLOAT_t,ndim=2] input_deltas
        if ada:
            # since sentences have different length, we keep a single ada.input
            for i in xrange(slen):
                ada.input[0] += np.square(grads.input[i])
            input_deltas = learning_rate * grads.input / np.sqrt(ada.input[0])
        else:
            input_deltas = grads.input * learning_rate
        
        padded_sentence = np.concatenate((self.pre_padding,
                                          sentence,
                                          self.post_padding))
        
        for i in xrange(slen):
            window = padded_sentence[i: i+window_size]
            self.converter.update(window, input_deltas[i])

        # cdef np.ndarray[INT_t,ndim=1] features
        # cdef np.ndarray[FLOAT_t,ndim=2] table
        # cdef int start, end, i, t

        # for i, deltas_i in enumerate(input_deltas):
        #     # deltas_i are input_deltas for i-th window in sentence
        #     # for each window (deltas_i: 300, features: 5)
        #     # this tracks where the deltas for the next table begins
        #     start = 0
        #     for features in padded_sentence[i:i+window_size]:
        #         # features for the i-th window
        #         # select the columns for each feature_tables
        #         for t, table in enumerate(self.feature_tables):
        #             end = start + table.shape[1]
        #             table[features[t]] += deltas_i[start:end]
        #             start = end
