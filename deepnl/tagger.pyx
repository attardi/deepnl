# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: profile=False

"""
Sequence tagger exploiting a neural network.
"""

# standard
import numpy as np
import cPickle as pickle
import sys                      # DEBUG

# local
import network
from networkseq import SequenceNetwork

# ----------------------------------------------------------------------

cdef class Tagger(object):
    """
    Abstract base class for sliding window sequence taggers.
    """
    
    # cdef dict tag_index         # tag ids
    # cdef list tags              # list of tags

    def __init__(self, nn, Converter converter, tag_index,
                 int left_context, int right_context):
        """
        :param nn: network to be used.
        :param converter: the Converter object that extracts features and
           converts them to weights.
        :param tag_index: index of tags.
        :param left_context: size of left context window.
        :param right_context: size of right context window.
        """
        self.nn = nn        # dependency injection
        self.converter = converter
        self.tag_index = tag_index
        self.tags = sorted(tag_index, key=tag_index.get)
        cdef np.ndarray padding_left = converter.get_padding_left()
        cdef np.ndarray padding_right = converter.get_padding_right()
        self.pre_padding = np.array(left_context * [padding_left])
        self.post_padding = np.array(right_context * [padding_right])

    def tag(self, sent):
        return self.tag_sequence(sent, return_tokens=True)

    cpdef tag_sequence(self, list tokens, bool return_tokens=False):
        """
        Tags a given list of tokens. 
        
        Tokens should be produced a compatible tokenizer in order to 
        match the entries in the vocabulary.
        
        :param tokens: a list of strings.
        :param return_tokens: whether to return also tokens.
        :returns: a list of tags or a list of pairs (token, tag) if
            :param return_tokens: is True.
        """
        cdef np.ndarray[INT_t,ndim=2] seq = self.converter.convert(tokens)
        # add padding
        seq = np.concatenate((self.pre_padding, seq, self.post_padding))

        cdef np.ndarray[FLOAT_t,ndim=2] scores = self._tag_sequence(seq)
        # computes full score, combining ftheta and A (if SLL)
        answer = self.nn._viterbi(scores)
        tags = [self.tags[tag] for tag in answer]

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
            Sentence includes padding.
        :param tags: the correct tags (needed when training).
        :return: an array of size (len(sentence), output_size) with the
            scores for all tags for each token..
        """
        nn = self.nn
        cdef int window_size = len(self.pre_padding) + 1 + len(self.post_padding)
        cdef slen = len(sentence) - window_size + 1 # without padding

        # scores[t, i] = ftheta_i,t = score for i-th tag, t-th word
        cdef np.ndarray[FLOAT_t,ndim=2] scores = np.empty((slen, nn.output_size))
        
        # container for network variables
        #vars = nn.variables()
        vars = network.Variables() # empty fields, filled below

        if train:
            # we must keep the whole history
            nn.input_sequence = np.empty((slen, nn.input_size))
            # hidden_values at each token in the correct path
            nn.hidden_sequence = np.empty((slen, nn.hidden_size))
        else:
            # we can discard intermediate values
            vars.input = np.empty(nn.input_size)
            vars.hidden = np.empty(nn.hidden_size)

        # print >> sys.stderr, sentence[:4,0]   # DEBUG
        # print >> sys.stderr, 'hweights', nn.p.hidden_weights[:4,:4] # DEBUG
        # print >> sys.stderr, 'hbias', nn.p.hidden_bias[:4]          # DEBUG

        # lookup the whole sentence at once
        # number of features in a window
        cdef int token_size = nn.input_size / window_size
        cdef np.ndarray sentence_features = np.empty(len(sentence) * token_size)
        self.converter.lookup(sentence, sentence_features)

        # run through all windows in the sentence
        cdef int i, start
        for i in xrange(slen):
            start = i * token_size
            vars.input = sentence_features[start: start+nn.input_size]
            if train:
                nn.input_sequence[i,:] = vars.input
                vars.hidden = nn.hidden_sequence[i]
            vars.output = scores[i]
            nn.forward(vars)
            # DEBUG
            # if train:
            #     print >> sys.stderr, 'window:', self.converter.extractors[0].sentence(window)
            #     print >> sys.stderr, 'sent:', window[:4], window[-4:]
            #     print >> sys.stderr, 'input', vars.input[:4], vars.input[-4:]
            #     print >> sys.stderr, 'iw', self.nn.p.hidden_weights[0,:4], self.nn.p.hidden_weights[-1,-4:]
            #     print >> sys.stderr, 'hidden', vars.hidden[:4], vars.hidden[-4:]
            #     print >> sys.stderr, 'output', vars.output[:4], vars.output[-4:]
        
        return scores

    def save(self, file):
        """
        Saves the tagger to a file.
        """
        self.nn.save(file)
        pickle.dump(self.tag_index, file)
        pickle.dump((len(self.pre_padding), len(self.post_padding)), file)
        self.converter.save(file)

    @classmethod
    def load(cls, file):
        """
        Loads the tagger from a file.
        """
        nn = SequenceNetwork.load(file)
        tag_index = pickle.load(file)
        left_context, right_context = pickle.load(file)
        converter = Converter()
        converter.load(file)

        return cls(nn, converter, tag_index, left_context, right_context)

