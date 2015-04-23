# -*- coding: utf-8 -*-
# distutils: language = c++

"""
Sequence tagger exploiting a neural network.
"""

# standard
import sys                      # DEBUG
import numpy as np
import cPickle as pickle

# local
import network
from networkseq import SequenceNetwork

# ----------------------------------------------------------------------

cdef class Tagger(object):
    """
    Abstract base class for sliding window sequence taggers.
    """
    
    # cdef dict tags_dict
    # cdef list itd

    def __init__(self, Converter converter, tags_dict,
                 int left_context, int right_context,
                 int hidden_size=0, int output_size=0, nn=None):
        """
        :param converter: the Converter object that extracts features and
           converts them to weights.
        :param tags_dict: dictionary of tags.
        :param left_context: size of left context window.
        :param right_context: size of right context window.
        :param nn: network to be used, instead of creating one.
        """
        self.converter = converter
        self.tags_dict = tags_dict
        self.itd = sorted(tags_dict, key=tags_dict.get)
        #self.feature_tables = [e.table for e in converter.extractors]

        if nn:
            self.nn = nn        # dependency injection
        else:
            # sum the number of features in all tables 
            input_size = converter.size()
            window_size = left_context + 1 + right_context
            input_size *= window_size
            self.nn = SequenceNetwork(input_size, hidden_size, output_size)

        self.padding_left = converter.get_padding_left()
        self.padding_right = converter.get_padding_right()
        self.pre_padding = np.array(left_context * [self.padding_left])
        self.post_padding = np.array(right_context * [self.padding_right])
        """
Traceback (most recent call last):
  File "/project/piqasso/tools/deepnl/bin/dl-pos.py", line 243, in <module>
    main()
  File "/project/piqasso/tools/deepnl/bin/dl-pos.py", line 212, in main
    trainer = create_trainer(args, converter, tags_dict)
  File "/project/piqasso/tools/deepnl/bin/dl-pos.py", line 55, in create_trainer
    args.hidden, tags_dict, args.verbose)
  File "deepnl/trainer.pyx", line 109, in deepnl.trainer.TaggerTrainer.__init__ (deepnl/trainer.cpp:3999)
  File "tagger.pyx", line 56, in deepnl.tagger.Tagger.__init__ (deepnl/tagger.cpp:2474)
        """

    def tag(self, sent):
        return self.tag_sequence(sent, return_tokens=True)

    def tag_sequence(self, list tokens, bool return_tokens=False):
        """
        Tags a given list of tokens. 
        
        Tokens should be produced a compatible tokenizer in order to 
        match the entries in the vocabulary.
        
        :param tokens: a list of strings
        :param return_tokens: whether to return also tokens
        :returns: a list of tags or a list of pairs (token, tag) if
            :param return_tokens: is True.
        """
        cdef np.ndarray[INT_t,ndim=2] converted = self.converter.convert(tokens)
        cdef np.ndarray[FLOAT_t,ndim=2] scores = self._tag_sequence(converted)
        # computes full score, combining ftheta and A (if SLL)
        answer = self.nn._viterbi(scores)
        tags = [self.itd[tag] for tag in answer]

        if return_tokens:
            return zip(tokens, tags)
        else:
            return tags

    cpdef np.ndarray[FLOAT_t,ndim=2] _tag_sequence(self,
                                                   np.ndarray sentence,
                                                   bool train=False):
        """
        Runs the network for each element in the sentence and returns 
        the scores for all possibile tag sequences.
        
        :param sentence: an array, where each row encodes a token.
        :param tags: the correct tags (needed when training)
        :return: a (len(sentence), output_size) array with the scores for all
            tags for each token
        """
        cdef slen = len(sentence)
        nn = self.nn
        # scores[t, i] = ftheta_i,t = score for i-th tag, t-th word
        cdef np.ndarray[FLOAT_t,ndim=2] scores = np.empty((slen, nn.output_size))
        
        if train:
            # we must keep the whole history
            nn.input_sequence = np.empty((slen, nn.input_size))
            # hidden_values at each token in the correct path
            nn.hidden_sequence = np.empty((slen, nn.hidden_size))
        else:
            # we can discard intermediate values
            nn.input_sequence = np.empty(nn.input_size)
            nn.hidden_sequence = np.empty(nn.hidden_size)
        
        # add padding to the sentence
        cdef np.ndarray[INT_t,ndim=2] padded_sentence = \
            np.concatenate((self.pre_padding,
                            sentence,
                            self.post_padding))
        cdef int window_size = len(self.pre_padding) + 1 + len(self.post_padding)
        cdef int i

        # container for network variables
        vars = network.Variables()

        #print >> sys.stderr, padded_sentence   # DEBUG
        #print >> sys.stderr, 'hweights', nn.hidden_weights[:4,:4] # DEBUG
        #print >> sys.stderr, 'hbias', nn.hidden_bias[:4]          # DEBUG
        # run through all windows in the sentence
        for i in xrange(slen):
            window = padded_sentence[i: i+window_size]
            if train:
                vars.input = nn.input_sequence[i]
                vars.hidden = nn.hidden_sequence[i]
            else:
                vars.input = nn.input_sequence
                vars.hidden = nn.hidden_sequence
            self.converter.lookup(window, vars.input)
            vars.output = scores[i]
            nn.run(vars)
            # DEBUG
            # if train:
                # print >> sys.stderr, 'input', vars.input[:4], vars.input[-4:]
                # print >> sys.stderr, 'hidden', vars.hidden[:4], vars.hidden[-4:]
                # print >> sys.stderr, 'output', vars.output[:4], vars.output[-4:]
        
        return scores

    cpdef update(self, SeqGradients grads, float learning_rate,
                 np.ndarray[INT_t,ndim=2] sentence, SeqGradients ada=None):

        # update network weights and transition weights.
        (<SequenceNetwork>self.nn)._update(grads, learning_rate, ada)
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

    def save(self, file):
        """
        Saves the tagger to a file.
        """
        self.nn.save(file)
        pickle.dump(self.tags_dict, file)
        pickle.dump((len(self.pre_padding), len(self.post_padding)), file)
        self.converter.save(file)

    @classmethod
    def load(cls, file):
        """
        Loads the tagger from a file.
        """
        nn = SequenceNetwork.load(file)
        tag_dict = pickle.load(file)
        (left_context, right_context) = pickle.load(file)
        converter = Converter()
        converter.load(file)

        return cls(converter, tag_dict, left_context, right_context, nn=nn)

