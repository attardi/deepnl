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
                 int hidden_size, int output_size=1,
                 Network nn=None, int ngrams=1):
        """
        Creates a neural network initialized for training.
        """
        
        self.converter = converter
        self.left_context = left_context
        self.right_context = right_context
        self.learning_rate = learning_rate
        self.ngram_size = ngrams
        self.avg_error = MovingAverage()
        self.verbose = False
        self.saver = lambda x: None

        if nn:
            self.nn = nn        # dependency injection
        else:
            # sum the number of features in all extractors' tables 
            input_size = (left_context + 1 + right_context) * converter.size()

            self.nn = Network(input_size, hidden_size, output_size)

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

    def __init__(self, converter, learning_rate, left_context, right_context,
                 hidden_size, tag_dict, verbose=False):
        self.tagger = Tagger(converter, tag_dict, left_context, right_context,
                             hidden_size, len(tag_dict))
        super(TaggerTrainer, self).__init__(converter, learning_rate,
                                            left_context, right_context,
                                            hidden_size, len(tag_dict),
                                            nn=self.tagger.nn)
        self.verbose = verbose

    def train(self, list sentences, list tags, 
              int epochs, int report_frequency=0,
              float desired_accuracy=0, threads=1):
        """
        Trains the network to tag sentences.
        
        :param sentences: a list of 2-dim numpy arrays, where each item 
            encodes a sentence. Each item in a sentence has the 
            indices to its features.
        :param tags: a list of 1-dim numpy arrays, where each item has
            the tags of the corresponding sentence.
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
            self._train_epoch(sentences, tags)
            
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
                if self.accuracy >= desired_accuracy > 0 \
                        or (self.nn.error > last_error and self.accuracy < last_accuracy):
                    break
                
            last_accuracy = self.accuracy
            last_error = self.nn.error

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
    
    def _train_epoch(self, list sentences, list tags):
        """
        Trains for one epoch with all examples.
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
        cdef SeqGradients ada
        ada = SeqGradients(nn.input_size, nn.hidden_size, nn.output_size, 1)
        #ada = None # DEBUG

        # keep last 2% for validation
        cdef int validation = int((len(sentences) - 1) * 0.98) + 1 # at least 1

        cdef int i = 0
        for sent, sent_tags in izip(sentences, tags):
            scores = self.tagger._tag_sequence(sent, True)
            grads = SeqGradients(nn.input_size, nn.hidden_size,
                                 nn.output_size, len(sent_tags))
            if nn._calculate_gradients_sll(sent_tags, grads, scores):
                nn._backpropagate(grads)
                self.tagger.update(grads, self.learning_rate, sent, ada)
            else:
                self.skips += 1

            self.train_items += len(sent)
            i += 1

            # progress report
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
        """Perform validation on held out data and estimate accuracy"""
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

