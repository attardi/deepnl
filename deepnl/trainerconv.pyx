# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: profile=False

"""
Train a DL Convolutional Neural Network.
"""

import sys
import numpy as np
from itertools import izip

# local
from networkconv cimport *
from trainer cimport Trainer
from extractors cimport Converter
from classifier import Classifier

# for decorations
cimport cython

cdef class ConvTrainer(Trainer):
    """
    Trainer for a convolutional network.
    """

    cdef readonly classifier

    def __init__(self, nn, Converter converter, list labels, dict options):
        """
        :param labels: list of labels.
        """
        super(ConvTrainer, self).__init__(nn, converter, options)
        left_context = options.get('left_context', 2)
        right_context = options.get('right_context', 2)
        self.classifier = Classifier(converter, labels, left_context, right_context, nn)
    
    def _train_epoch(self, list sentences, list labels):
        """
        Trains for one epoch with all examples.
        :param sentences: a list of 2-dim numpy arrays, where each array 
            encodes a sentence. Each array row represents a token through the
            indices to its features.
        :param labels: a list of id of labels for each corresponding sentence.
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
        cdef ConvGradients grads
        cdef int_t i = 0, slen
        for i in xrange(validation):
            sent = sentences[i]
            label = labels[i]
            slen = len(sent)
            # add padding
            sent = np.concatenate((self.pre_padding, sent, self.post_padding))
            vars = nn.variables(len(sent)) # allocate variables
            self.converter.lookup(sent, vars.input)
            nn.forward(vars)
            grads = nn.gradients(slen) # allocate gradients
            loss = nn.backpropagate(label, vars, grads)
            if loss > 0.0:
                self.error += loss
                self.update(grads, sent)
                # # DEBUG. verify
                # nn.forward(vars) # DEBUG
                # loss2 = nn.backpropagate(label, vars, grads) # DEBUG
                # if loss2 > loss:                             # DEBUG
                #     print >> sys.stderr, 'WORSE', i, label, loss, loss2 # DEBUG
            else:
                self.epoch_hits += 1
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
        cdef int_t count = 0
        cdef int_t hits = 0

        cdef int_t i, label
        cdef np.ndarray[int_t,ndim=2] sent
        cdef Variables vars

        for i in xrange(idx, len(sentences)):
            sent = sentences[i]
            label = labels[i]
            # add padding
            sent = np.concatenate((self.pre_padding, sent, self.post_padding))
            vars = self.nn.variables(len(sent)) # allocate variables
            self.converter.lookup(sent, vars.input)
            self.nn.forward(vars)
            if np.argmax(vars.output) == label:
                hits += 1
            count += 1
            if self.verbose:
                if (i+1-idx)%1000 == 0:
                    sys.stderr.write('+')
                    sys.stderr.flush()
                elif (i+1-idx)%100 == 0:
                    sys.stderr.write('.')
                    sys.stderr.flush()
        if self.verbose:
            sys.stderr.write('\n')
        return float(hits) / count if count else 0.0

    # def save(self, file): inherited

    @classmethod
    def load(cls, file):
        """
        Resume training from previous dump.
        """
        # use __new__() to skip initialiazation
        trainer = ConvTrainer.__new__(cls) # CHECKME: ConvTrainer is redundant?
        trainer.pre_padding, trainer.post_padding, trainer.ngram_size = np.load(file)
        trainer.nn = ConvolutionalNetwork.load(file) # different from super
        trainer.converter = Converter()
        trainer.converter.load(file)
        return trainer
