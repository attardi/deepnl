# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: profile=True

"""
Train a DL Neural Network.
"""

import numpy as np
from numpy import int32 as INT
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

    cdef add(self, float_t v):
        """Add value :param v: to the moving average."""
        self.count += 1
        self.mean += (2. / self.count) * (v - self.mean)
        cdef float_t this_variance = (v - self.mean) * (v - self.mean)
        self.variance += (2. / self.count) * (this_variance - self.variance)

# ----------------------------------------------------------------------

cdef float_t skipErr = 0.01          # minimum error worth an update

cdef class Trainer(object):
    """
    Abstract class for trainers.
    """

    max_no_progress = 3         # stop after these epochs without progress

    # cdef np.ndarray pre_padding, post_padding
    # size of ngrams
    # cdef int_t ngram_size
    # cdef public float_t learning_rate
    # cdef public object saver
    # cdef int_t total_items, epoch_items, epoch_hits, skips
    # data for statistics
    # cdef float_t error, accuracy
    # cdef readonly MovingAverage avg_error
    # cdef public bool verbose
    
    def __init__(self, Network nn, Converter converter, dict options):
        """
        Creates a neural network initialized for training.
        :param nn: the network to be trained.
        :param converter: feature extractor.
        :param options: optional parameters.
        """
        
        self.nn = nn        # dependency injection
        self.converter = converter
        self.avg_error = MovingAverage()
        self.saver = lambda x: None

        # options
        self.learning_rate = options.get('learning_rate', 0.01)
        self.adaEps = options.get('eps', 1e-6) # 1e-8
        self.adaRo = options.get('ro', 0.95)
        self.l1_decay = options.get('l1_decay', 0.0)
        self.l2_decay = options.get('l2_decay', 0.0)
        self.momentum = options.get('momentum', 0.9)
        self.skipErr = skipErr     # skip errors < this value

        self.verbose = options.get('verbose', False)
        self.ngram_size = options.get('ngram_size', 1)

        left_context = options.get('left_context', 2)
        right_context = options.get('right_context', 2)
        cdef np.ndarray padding_left = converter.get_padding_left()
        cdef np.ndarray padding_right = converter.get_padding_right()
        self.pre_padding = np.array(left_context * [padding_left], dtype=INT)
        self.post_padding = np.array(right_context * [padding_right], dtype=INT)

    def train(self, list examples, list outcomes, 
              int_t epochs, int_t report_frequency=0,
              float_t desired_accuracy=0, threads=1):
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
        logger.info("Parameters: lr=%f, eps=%.2e, ro=%f" %
                    (self.learning_rate, self.adaEps, self.adaRo))
        logger.info("Training for up to %d epochs" % epochs)
        
        # prepare for AdaGrad
        if self.adaEps:
            self.ada = self.nn.parameters()
            self.ada = self.ada.clear(self.adaEps)
            self.converter.adaGradInit(self.adaEps)
        else:
            self.ada = None

        self.total_items = 0
        cdef float_t top_accuracy = 0
        cdef float_t last_accuracy = 0
        cdef float_t min_error = np.Infinity 
        cdef float_t last_error = np.Infinity
        
        cdef int_t no_progress = 0 # how many epochs without progress

        np.seterr(all='raise')

        for i in xrange(epochs):
            self._train_epoch(examples, outcomes)

            self.total_items += self.epoch_items
            
            # normalize error
            self.error /= self.epoch_items #if self.epoch_items else np.Infinity
            if self.accuracy > top_accuracy:
                top_accuracy = self.accuracy
            
            if (report_frequency > 0 and i % report_frequency == 0) \
                or self.accuracy >= desired_accuracy > 0 \
                or (self.accuracy < last_accuracy and self.error > last_error):
                self._epoch_report(i + 1)

            # early stop
            if self.accuracy >= desired_accuracy > 0:
                break
            if self.error > last_error and self.accuracy < last_accuracy:
                no_progress += 1
            if no_progress == Trainer.max_no_progress:
                break
            no_progress = 0

            # save model
            if self.error < min_error:
                min_error = self.error
                logger.info("Saving model...")
                self.saver(self)

            last_accuracy = self.accuracy
            last_error = self.error

    def _train_epoch(self, list sentences, list labels):
        """
        Trains for one epoch with all examples.
        :param sentences: a list of 2-dim numpy arrays, where each array 
            encodes a sentence. Each array row represents a token through the
            indices to its features.
        :param labels: a list of id of labels of each corresponding sentence.
        """

        self.error = 0
        self.epoch_items = 0
        self.epoch_hits = 0
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
        vars = nn.variables()   # allocate variables
        cdef int_t i = 0
        for sent, label in izip(sentences, labels):
            # add padding
            sent = np.concatenate((self.pre_padding, sent, self.post_padding))
            self.converter.lookup(sent, vars.input)
            nn.forward(vars)
            grads = nn.gradients(len(sent)) # allocate gradients
            loss = nn.backpropagate(label, vars, grads)
            if loss > self.skipErr:
                self.error += loss
                self.update(grads, sent)
            else:
                self.skips += 1
            self.epoch_items += 1
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
    cdef float_t _validate(self, list sentences, labels, int_t idx):
        """Perform validation on held out data and estimate accuracy
        :param idx: index of first sentence in validation set.
        """
        cdef int_t tokens = 0
        cdef int_t hits = 0

        cdef int_t i, label
        cdef np.ndarray[int_t,ndim=2] sent
        cdef Variables vars = self.nn.variables()

        for i in xrange(idx, len(sentences)):
            sent = sentences[i]
            label = labels[i]
            # add padding
            sent = np.concatenate((self.pre_padding, sent, self.post_padding))
            self.converter.lookup(sent, vars.input)
            self.nn.forward(vars)
            if np.argmax(vars.output) == label:
                hits += 1
            tokens += 1
        if tokens:
            return float(hits) / tokens
        else:
            return 0.0

    def _epoch_report(self, int_t num):
        """
        Reports the status of the network in the given training epoch,
        including error and accuracy.
        """
        logger = logging.getLogger("Logger")
        msg = ""
        if self.skips:
            msg = "   %d corrections skipped" % self.skips
        logger.info("%d epochs   Examples: %d   Error: %f   Accuracy: %f%s" \
                        % (num, self.total_items, self.error, self.accuracy, msg))

    cpdef update(self, Gradients grads, np.ndarray[int_t,ndim=2] sequence):
        """
        Adjust the weights.
        :param grads: gradients to apply.
        :param sequence: portion of padded sentence.
        """

        # update network weights.
        self.nn.update(grads, self.learning_rate, self.ada)

        self.converter.update(grads.input, sequence,
                              self.learning_rate)

    def save(self, file):
        np.save(file, (self.pre_padding, self.post_padding, self.ngram_size))
        self.nn.save(file)
        self.converter.save(file)

    def save_vectors(self, file, variant):
        self.converter.extractors[0].save_vectors(file, variant)

    @classmethod
    def load(cls, file):
        """
        Resume training from previous dump.
        """
        # use __new__() to skip initialiazation
        trainer = __new__(cls)
        trainer.pre_padding, trainer.post_padding, trainer.ngram_size = np.load(file)
        trainer.nn = Network.load(file)
        trainer.converter = Converter()
        trainer.converter.load(file)
        return trainer

# ----------------------------------------------------------------------

cdef class TaggerTrainer(Trainer):
    """
    A trainer for sequence taggers.
    """

    def __init__(self, nn, Converter converter, tag_index, options):
        super(TaggerTrainer, self).__init__(nn, converter, options)
        left_context = options.get('left_context', 2)
        right_context = options.get('right_context', 2)
        self.tagger = Tagger(nn, converter, tag_index, left_context, right_context)

    def _train_epoch(self, list sentences, list tags):
        """
        Trains for one epoch with all examples.
        :param sentences: a list of 2-dim numpy arrays, where each array 
            encodes a sentence. Each array row represents a token through the
            indices to its features.
        :param tags: a list of 1-dim numpy arrays, where each item has
            the tags of the corresponding sentence.
        """
        # FIXHIM: should be a parametric type
        cdef SequenceNetwork nn = <SequenceNetwork>self.tagger.nn     # same as self.nn
        self.error = 0
        self.epoch_items = 0
        self.epoch_hits = 0
        self.skips = 0
        
        # shuffle data
        # get the random number generator state in order to shuffle
        # sentences and their tags in the same order
        random_state = np.random.get_state()
        np.random.shuffle(sentences)
        np.random.set_state(random_state)
        np.random.shuffle(tags)
        
        cdef SeqGradients grads

        # keep last 2% for validation
        cdef int_t validation = int((len(sentences) - 1) * 0.98) + 1 # at least 1

        cdef float_t error
        cdef int_t i = 0, slen
        cdef np.ndarray[int_t,ndim=2] sent
        for sent, sent_tags in izip(sentences, tags):
            slen = len(sent)    # sequence length
            # add padding
            sent = np.concatenate((self.pre_padding, sent, self.post_padding))
            scores = self.tagger._tag_sequence(sent, True)
            grads = nn.gradients(slen) # one for each item
            error = nn.backpropagateSeq(sent_tags, scores, grads, self.skipErr)
            if error > self.skipErr:
                self.error += error
                self.update(grads, sent)
            else:
                self.skips += 1

            self.epoch_items += slen

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
        # print('hw', nn.hidden_weights[:4,:4], file=sys.stderr)
        # print('ow', nn.output_weights[0:4,:4], file=sys.stderr)

    @cython.boundscheck(False)
    cdef float_t _validate(self, list sentences, tags, int_t idx):
        """Perform validation on held out data and estimate accuracy
        :para sentences: full training set.
        :para tags: tags for all sentences.
        :param idx: index of first sentence in validation set.
        """
        cdef int_t tokens = 0
        cdef int_t hits = 0

        cdef np.ndarray[int_t,ndim=2] sent
        cdef np.ndarray[int_t] answer
        cdef np.ndarray[float_t,ndim=2] scores
        cdef int_t i, pred_tag, gold_tag

        for i in xrange(idx, len(sentences)):
            sent = sentences[i]
            gold_tags = tags[i]
            # add padding
            sent = np.concatenate((self.pre_padding, sent, self.post_padding))
            scores = self.tagger._tag_sequence(sent)
            answer = self.tagger.nn._viterbi(scores)
            for pred_tag, gold_tag in izip(answer, gold_tags):
                if pred_tag == gold_tag:
                    hits += 1
                tokens += 1
        return float(hits) / tokens if tokens else 0.0

    cpdef update(self, Gradients grads, np.ndarray[int_t,ndim=2] sentence):

        """
        :param grads: gradients to apply.
        :param sentence: the padded sentence.
        """
        # update network weights and transition weights.
        self.nn.update(grads, self.learning_rate, self.ada)
        #
        # Adjust the features indexed by the input windows.
        #
        # the deltas that will be applied to the feature tables
        # they are in the same sequence as the network receives them, i.e.
        # [token1-table1][token1-table2][token2-table1][token2-table2] (...)
        # e.g. num features = 50 (embeddings) + 5 (caps) + 5 (suffix) = 60
        # input_size = num features * window (e.g. 60 * 5).

        cdef int_t window_size = len(self.pre_padding) + 1 + len(self.post_padding)
        cdef int_t i, slen = len(sentence) - window_size + 1 # without padding

        # DEBUG: Old version
        # if ada:
        #     # accumulate in first position
        #     for i in xrange(slen):
        #         ada.input[0] += np.square(grads.input[i])
        #     grads.input /= np.sqrt(ada.input[0] + adaEps)

        cdef np.ndarray[int_t,ndim=2] window
        for i in xrange(slen):
            window = sentence[i: i+window_size]
            self.converter.update(grads.input[i], window, self.learning_rate)
