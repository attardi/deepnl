# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: profile=True

"""
Train a DL Convolutional Neural Network.
"""

import sys
import numpy as np
from itertools import izip

# local
from networkconv cimport *
from trainer cimport Trainer

# for decorations
cimport cython

cdef class ConvTrainer(Trainer):
    """
    Trainer for a convolutional network.
    """

    def __init__(self, nn, Converter converter, dict options):
        super(ConvTrainer, self).__init__(nn, converter, options)
    
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
        ada = nn.gradients()
        cdef int i = 0
        for i in xrange(validation):
            sent = sentences[i]
            label = labels[i]
            # add padding
            sent = np.concatenate((self.pre_padding, sent, self.post_padding))
            vars = nn.variables(len(sent)) # allocate variables
            self.converter.lookup(sent, vars.input)
            nn.forward(vars)
            grads = nn.gradients(len(sent)) # allocate gradients
            loss = nn.backpropagate(label, vars, grads)
            if loss > self.skipErr:
                self.error += loss
                self.update(grads, self.learning_rate, sent, ada)
                # DEBUG. verify
                nn.forward(vars)
                loss2 = nn.backpropagate(label, vars, grads)
                if loss2 > loss:
                    print >> sys.stderr, i, loss, loss2
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
    cdef float _validate(self, list sentences, labels, int idx):
        """Perform validation on held out data and estimate accuracy
        :param idx: index of first sentence in validation set.
        """
        cdef int count = 0
        cdef int hits = 0

        cdef int i, label
        cdef np.ndarray[INT_t,ndim=2] sent
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
        trainer = ConvTrainer.__new__(cls)
        trainer.pre_padding, trainer.post_padding, trainer.ngram_size = np.load(file)
        trainer.nn = ConvolutionalNetwork.load(file)
        trainer.converter = Converter()
        trainer.converter.load(file)
        return trainer
