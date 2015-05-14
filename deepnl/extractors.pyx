# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: profile=False

"""
Feature extractors.

Each extractor maintains its own table, for mapping features to vectors.
Some of them also maintain a dictionary, for mapping tokens to features.
They are resposible for loading and saving these data to/from a model file.
"""

# standard
import logging
import numpy as np
import os
import re
from collections import Counter, defaultdict
import cPickle as pickle
from itertools import izip
import sys                      # modules

# local
from word_dictionary import WordDictionary as WD
import embeddings


# ----------------------------------------------------------------------

cdef class Iterable:
    """
    ABC for classes that provide the __iter__() method.
    """

    def __iter__(self):
        return self

# ----------------------------------------------------------------------

cdef class ConvertGenerator(Iterable):
    cdef Converter converter
    cdef sentences
    cdef converted
    cdef bool cache

    def __init__(self, converter, sentences, cache=False):
        """
        :param sentences: an iterable over sentences.
        :param cache: if this is True, caches converted sentences,
        avoiding repeated conversion. Useful if sentences is small enough to
        stay in memory.
        """
        self.converter = converter
        self.sentences = sentences
        self.converted = []     # cache converted sentences
        self.cache = cache

    def __iter__(self):
        cdef np.ndarray c
        if self.converted:
            for s in self.converted:
                yield s
        else:
            for s in self.sentences:
                c =  self.converter.convert(s)
                if self.cache:
                    self.converted.append(c)
                yield c

cdef class Converter(object):
    """
    Interface to the extractors.
    Extracts features from a sentence and converts them into a list of feature
    vectors in feature space.
    """
    
    # one extractor for each feature
    # cpdef list extractors

    def __init__(self):
        self.extractors = []
    
    cpdef int size(self):
        #return sum(e.size() for e in self.extractors)
        cdef int s = 0
        for e in self.extractors:
            s += e.size()
        return s

    def add(self, extractor):
        """
        Adds an extractor function to the SentenceConverter.
        In order to get a token's feature indices, the Converter will call
        each of its extraction functions passing the token as a parameter.

        """
        self.extractors.append(extractor)
    
    def generator(self, sentences, cache=False):
        """
        :return: a generator for converting the :param sentences:.
        """
        return ConvertGenerator(self, sentences, cache)

    cdef np.ndarray[INT_t,ndim=1] get_padding_left(self):
        """
        :return: the features of the left padding token for the sentence.
        
        """
        pad = WD.padding_left
        return self.convert([pad])[0] # features of first token
    
    cdef np.ndarray[INT_t,ndim=1] get_padding_right(self):
        """
        :return: the features of the right padding token for the sentence.
        """
        pad = WD.padding_right
        return self.convert([pad])[0] # features of first token
    
    cpdef np.ndarray[INT_t,ndim=2] convert(self, list sent):
        """
        Converts a sentence into an array of feature indices.
        :param sent: a list of words.
        :return: an array of all extractors' results.
        """
        return np.array(zip(*[extractor.extract(sent) for extractor in self.extractors]))
        # CHECKME: is this faster?
        # return np.array([extractor.extract(sent) for extractor in self.extractors]).T

    cpdef np.ndarray[FLOAT_t,ndim=1] lookup(self,
                                            np.ndarray[INT_t,ndim=2] sentence,
                                            np.ndarray out=None):
        """
        Collect the feature vectors of all sentence tokens.
        :param sentence: Each row represents a token through its indices into
            each feature table.
        :param out: vector where to store result.
        :return: a single feature vector, combining the vectors of all features
            of eack token in :param sentence:
        """
        if out is None:
            out = np.empty(self.size())
        cdef int start = 0, end
        for token in sentence:
            for feature, extractor in izip(token, self.extractors):
                end = start + extractor.size()
                np.copyto(out[start:end], extractor[feature])
                start = end
        return out

    cpdef update(self, np.ndarray[INT_t,ndim=2] sentence,
                 np.ndarray[FLOAT_t,ndim=1] grads):
        """
        Update the features according to the given gradients.
        :param sentence: Each row represents a token through its indices into
            each feature table.
        :param grads: vector of feature gradients.
        """
        cdef int start = 0, end
        for token in sentence:
            for feature, extractor in izip(token, self.extractors):
                end = start + extractor.size()
                extractor.table[feature] += grads[start:end] # __setitem__()
                start = end

    def save(self, file):
        """
        Save all extractors' data to file.
        """
        # save class names
        class_names = [type(e).__name__ for e in self.extractors]
        pickle.dump(class_names, file)
        for extractor in self.extractors:
            extractor.save(file)

    def load(self, file):
        """
        Load all extractors' data from file.
        """
        class_names = pickle.load(file)
        # FIXME: this will recognize only classes defined in this file
        #m = __import__("extractors")
        m = sys.modules['deepnl.extractors']
        for c in class_names:
            cls = getattr(m, c)
            self.add(cls.__new__(cls))
        for extractor in self.extractors:
            extractor.load(file)

# ----------------------------------------------------------------------

cdef class Extractor(object):
    """
    Abstract feature extractor.

    Each extractor deals with one kind of features, e.g. embeddings,
    pos, caps, etc.
    Each one is responsible for saving and loading its own data to a
    single model file.
    """

    # cpdef readonly dict dict
    # cpdef readonly np.ndarray table

    def __getitem__(self, feature):
        """
        Get the vector corresponding to the :param feature:
        """
        return self.table[feature]

    def __setitem__(self, feature, value):
        """
        Set the vector corresponding to the :param feature:
        """
        self.table[feature] = value

    cpdef int size(self):
        """
        :return: dimension of embeddings space
        """
        return self.table.shape[1]

    def save(self, file):
        pickle.dump(self.dict, file)
        pickle.dump(self.table, file)

    def load(self, file):
        self.dict = pickle.load(file)
        self.table = pickle.load(file)

# ----------------------------------------------------------------------

cdef class Embeddings(Extractor):
    """
    Lookup layer.
    """

    def __init__(self, size=0, vocab_file=None, vectors=None, vocab=None,
                 variant=None):
        """
        Construct from either precomputed vocabulary and vectors files,
        or from a list of words :param vocab:.
        :param size: vector space dimension.
        :param vocab_file: file containing the vocabulary
        :param vectors: file containing the vectors
        :param variant: style of embeddgins (senna, polyglot, word2vect)
        """
        if vocab:
            self.dict = <dict>WD(None, wordlist=vocab, variant=variant)
            self.table = embeddings.generate_vectors(len(self.dict), size)
        elif vocab_file:
            self.dict = <dict>WD(None, wordlist=self.load_vocabulary(vocab_file), variant=variant)
            if vectors and os.path.exists(vectors):
                self.table = self.load_vectors(vectors)
            else:
                self.table = embeddings.generate_vectors(len(self.dict), size)
        elif variant == 'word2vec':
            # load both vocab and vectors from single file
            self.table, wordlist = embeddings.Word2Vec.load(vectors)
            self.dict = <dict>WD(None, wordlist=wordlist, variant=variant)
            # add vectors for special symbols
            extra = len(self.dict) - len(self.table)
            self.table = np.concatenate((self.table, embeddings.generate_vectors(extra, self.table.shape[1])))

    def save(self, file):
        self.dict.save(file)
        pickle.dump(self.table, file)

    def load(self, file):
        self.dict = <dict>WD.load(file)
        self.table = pickle.load(file)

    def load_vocabulary(self, file, variant=None):
        # FIXME: allow chosing variant
        return embeddings.Plain.read_vocabulary(file)

    def save_vocabulary(self, file):
        # FIXME: allow chosing variant
        return embeddings.Plain.write_vocabulary(self.dict.keys(), file)

    def load_vectors(self, file, variant=None):
        # FIXME: allow choosing variant
        return embeddings.Plain.read_vectors(file)

    def save_vectors(self, file):
        return embeddings.Plain.write_vectors(file, self.table)

    def extract(self, words):
        """
        Extract the features representing each word.
        """
        return [self.dict[token] for token in words]

    def lookup_ngram(self, ngramIDs):
        """
        Lookup an ngrams.
        """
        # lookup ngram IDs to obtain back words
        tokens = self.dict.get_words(ngramIDs)
        return self.dict[' '.join(tokens)]

# ----------------------------------------------------------------------
# Capitalization

cdef class Caps(object):                     # Caps(Enumeration)
    """
    Enumeration for capitalization types.
    """
    # lower = 0
    # title = 1
    # non_alpha = 2
    # other = 3
    # upper = 4                   # Attardi
    # num_values = 5

    # SENNA
    padding = 0
    upper  = 1
    hascap = 2
    title  = 3
    nocaps = 4
    num_values = 5              # extractor values

cdef class CapsExtractor(Extractor):

    def __init__(self, size):
        self.table = embeddings.generate_vectors(Caps.num_values, size)

    @staticmethod
    def type(word):
        """
        Returns a code describing the capitalization of the word:
        lower, title, upper, other or non-alpha (numbers and other tokens that can't be
        capitalized).
        """
        # if not word.isalpha():
        #     return Caps.non_alpha

        # if word.istitle():
        #     return Caps.title

        # if word.islower():
        #     return Caps.lower

        # if word.isupper():          # Attardi
        #     return Caps.upper

        # return Caps.other

        # SENNA
        if word == WD.padding_left or word == WD.padding_right:
            return Caps.padding

        if word.isupper():
            return Caps.upper

        if word[0].isupper():       # istitle() checks other letters too
            return Caps.title

        # can't use islower() because it accepts '3b'
        for c in word:
            if c.isupper():
                return Caps.hascap

        return Caps.nocaps

    def extract(self, words):
        """
        :return: the list of capitalization codes for the given list of words.
        """
        return map(CapsExtractor.type, words)

    def save(self, file):
        # no dictionary
        pickle.dump(self.table, file)

    def load(self, file):
        self.table = pickle.load(file)

def capitalize(word, capitalization):
    """
    Capitalizes the word in the desired format. If the capitalization is 
    Caps.other, it is set all uppercase.
    """
    if capitalization == Caps.non_alpha:
        return word
    elif capitalization == Caps.lower:
        return word.lower()
    elif capitalization == Caps.title:
        return word.title()
    elif capitalization == Caps.other:
        return word.upper()
    else:
        raise ValueError("Unknown capitalization type.")

# ----------------------------------------------------------------------
    
cdef class AffixExtractor(Extractor):
    """Abstract class for prefix or suffix extractors."""

    other = 1                   # NOSUFFIX
    padding = 0
    specials = 2                # number of specials (other, padding)

    def __init__(self, size, filename=None, wordlist=[]):
        """
        :param size: the dimension of the embeddings space
        """
        specials = AffixExtractor.specials
        if filename:
            self.load_affixes(filename)
        else:
            affixes = self.build(wordlist)
            # leave reserved values for specials
            self.dict = { x: i+specials for i,x in enumerate(affixes) }
        # create vectors for possible values
        self.table = embeddings.generate_vectors(len(self.dict)+specials, size)

    def extract(self, words):
        """
        :return: the list of prefix/suffix codes for the given :param words:.
        """
        return [self.affix(w) for w in words]

    def load_affixes(self, filename):
        """
        Load prefixes or suffixes from file :param filename:.
        """
        logger = logging.getLogger("Logger")
        specials = AffixExtractor.specials
        try:
            with open(filename, 'rb') as f:
                self.dict = {}
                for line in f:
                    affix = unicode(line.strip(), 'utf-8')
                    # leave reserved values for specials
                    self.dict[affix] = len(self.dict) + specials
        except IOError:
            logger.error("File %s doesn't exist." % filename)
            raise

# ----------------------------------------------------------------------

cdef class SuffixExtractor(AffixExtractor):

    # max suffix length (mimic SENNA)
    max_length = 2

    def affix(self, word):
        """
        Return the suffix code for the given word.
        """
        if word == WD.padding_left or word == WD.padding_right:
            return AffixExtractor.padding

        # as in SENNA, we use the whole word in this case:
        # if len(word) <= SuffixExtractor.max_length:
        #     return Affix.other

        suffix = word[-SuffixExtractor.max_length:].lower()
        return self.dict.get(suffix, AffixExtractor.other)

    def build(self, wordlist, num=200, min_occurrences=3,
              size=SuffixExtractor.max_length):
        """
        Creates a list with the most common suffixes found in 
        wordlist.
        
        :param wordlist: a list of words (there shouldn't be repetitions)
        :param num: maximum number of suffixes
        :param min_occurrences: minimum number of occurrences of each suffix
        in wordlist
        :param size: desired size of suffixes
        """
        # we take the whole word if len(x) <= size
        all_endings = (x[-size:] for x in wordlist 
                       if not re.search('_|\d', x[-size:]))
        c = Counter(all_endings)
        common_endings = c.most_common(num)
        return [e for e, n in common_endings if n >= min_occurrences]

# ----------------------------------------------------------------------

cdef class PrefixExtractor(AffixExtractor):

    # max prefix length (mimic SENNA)
    max_length = 2

    def affix(self, word):
        """
        Return the prefix code for the given :param word:.
        """
        if word == WD.padding_left or word == WD.padding_right:
            return AffixExtractor.padding

        # as in SENNA, we use the whole word in this case:
        # if len(word) <= PrefixExtractor.max_length:
        #     return Affix.other

        prefix = word[:PrefixExtractor.max_length].lower()
        return self.dict.get(prefix, AffixExtractor.other)

    def build(cls, wordlist, num=200, min_occurrences=3,
              size=PrefixExtractor.max_length):
        """
        Creates a list with the most common prefixes found in 
        wordlist.
        
        :param wordlist: a list of words (there shouldn't be repetitions)
        :param num: maximum number of prefixes
        :param min_occurrences: minimum number of occurrences of each prefix
        in wordlist
        :param size: desired size of prefixes
        """
        # we take the whole word if len(x) <= size
        all_beginnings = (x[-size:] for x in wordlist 
                       if  not re.search('_|\d', x[-size:]))
        c = Counter(all_beginnings)
        common_beginnings = c.most_common(num)
        return [e for e, n in common_beginnings if n >= min_occurrences]

# ----------------------------------------------------------------------

cdef class GazetteerExtractor(Extractor):

    absent = 0
    present = 1
    padding = 2
    num_values = 3              # extractor values

    def __init__(self, words=None, size=5, lowcase=True):
        """
        :param words: list of words to add to gazeetteer.
        :param size: vector dimension.
        :param lowcase: whether to compare lowercase words.
        """
        self.lowcase = lowcase
        if words:
            self.dict = {x: GazetteerExtractor.present for x in words}
            self.table = embeddings.generate_vectors(GazetteerExtractor.num_values, size)

    def extract(self, words):
        """
        Check presence in dictionary possibly as multiword.
        Set to 'present' items corresponding to words present in dictionary
        :return: the list of codes for the given :param words:.
        """
        res = [GazetteerExtractor.absent] * len(words)
        for i, token in enumerate(words):
            if token == WD.padding_left or token == WD.padding_right:
                res[i] = GazetteerExtractor.padding
                continue
            entity = token.lower() if self.lowcase else token
            if entity in self.dict:
                res[i] = GazetteerExtractor.present
            for j in range(i+1, len(words)):
                if self.lowcase:
                    entity += ' ' + words[j].lower()
                else:
                    entity += ' ' + words[j]
                if entity in self.dict:
                    for k in range(i, j+1):
                        res[k] = GazetteerExtractor.present
        return res

    @classmethod
    def create(cls, filename, size=5, lowcase=True):
        """
        Create extractors from gazeeteer file, consisting of lines:
          TYPE\tentity
        :param filename: file where to read list.
        :param size: size of vectors to generate.
        """
        classes = {}
        with open(filename) as file:
            for line in file:
                line = line.strip().decode('utf-8')
                c, words = line.split(None, 1)
                if lowcase:
                    words = words.lower()
                if c not in classes:
                    classes[c] = set()
                classes[c].add(words)
        extractors = [cls(w, size, lowcase) for w in classes.values()]
        return extractors

    def save(self, file):
        pickle.dump(self.dict, file)
        pickle.dump(self.table, file)
        pickle.dump(self.lowcase, file)

    def load(self, file):
        self.dict = pickle.load(file)
        self.table = pickle.load(file)
        self.lowcase = pickle.load(file)

# ----------------------------------------------------------------------

cdef class AttributeExtractor(Extractor):
    """
    Extract a token attribute as feature.
    """

    padding = 0

    def __init__(self, idx, size=5):
        """
        :param idx: index of token attribute to use.
        :[aram size: vector dimension.
        """
        self.idx = idx
        self.table = embeddings.generate_vectors(AttributeExtractor.num_values, size)

    def extract(self, words):
        """
        :return: the list of POS codes for the given :param words:.
        """
        res = [0] * len(words)
        for i, token in enumerate(words):
            if token == WD.padding_left or token == WD.padding_right:
                res[i] = AttributeExtractor.padding
                continue
            attr = token[self.idx]
            if attr not in self.dict:
                self.dict[attr] = len(self.dict)
            res[i] = self.dict[attr]
        return res

    def save(self, file):
        pickle.dump(self.dict, file)
        pickle.dump(self.table, file)
        pickle.dump(self.lowcase, file)

    def load(self, file):
        self.dict = pickle.load(file)
        self.table = pickle.load(file)
        self.lowcase = pickle.load(file)
