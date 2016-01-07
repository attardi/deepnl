# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict
import cPickle as pickle
import re
from numpy import int32 as INT

num = re.compile('[+\-]?([0-9][,.]?)+$')

def isNumber(key):
    return num.match(key)

class WordDictionary(dict):
    """
    Class to store words and their corresponding indices in
    the network lookup table. Also deals with padding and
    maps rare words to a special index.
    """
    
    def __init__(self, sentences, size=None, minimum_occurrences=None,
                 wordlist=None, variant='senna'):
        """
        Fills a dictionary (to be used for indexing) with the most
        common words in the given text.
        
        :param sentences: an iterable on lists of tokens 
            (each token represented as a string).
        :param size: Maximum number of token indices 
            (not including paddings, rare, etc.).
        :param minimum_occurrences: The minimum number of occurrences a token
            must have in order to be included.
        :param wordlist: Use this list of words to build the dictionary.
            Overrides sentences if not None and ignores maximum size.
        :param variant: either 'polyglot', 'word2vec', or 'senna' conventions,
            i.e. keep case, use different padding tokens.
        """
        self.variant = variant
        if variant:
            self.variant = variant.lower()
        if self.variant == 'polyglot':
            padding_left = '<PAD>'
            padding_right = '<PAD>'
            rare = '<UNK>'
        elif self.variant == 'word2vec':
            padding_left = '</s>'
            padding_right = '</s>'
            rare = '<UNK>'
        elif self.variant == 'senna':
            # SENNA conventions
            padding_left = 'PADDING'
            padding_right = 'PADDING'
            rare = 'UNKNOWN'
        
        if self.variant:
            self.special_symbols = set((rare, 
                                        padding_left,
                                        padding_right))
        else:
            self.special_symbols = set()

        if wordlist is None:
            # work with the supplied sentences. extract frequencies.
            
            # gets frequency count
            c = self._get_frequency_count(sentences)
        
            if minimum_occurrences is None:
                minimum_occurrences = 1
            
            words = [key for key, number in c.most_common() 
                     if number >= minimum_occurrences]
            
            if size is not None and size < len(words):
                words = words[:size]
        
        else:
            # Keep the order and eliminate duplicates
            #words = list(OrderedDict.fromkeys(self.normalize(w) for w in wordlist))
            words = list(OrderedDict.fromkeys(wordlist))
            
        # trim to the maximum size
        if size is None:
            size = len(words)
        else:
            size = min(size, len(words))
            words = words[:size]
        
        # build the inverse index
        self.words = [0] * len(words) # inverse index
        for idx, word in enumerate(words):
            super(WordDictionary, self).__setitem__(word, INT(idx))
            self.words[idx] = word # words should be already normalized
        
        # if the given words include one of the rare or padding symbols,
        # don't replace it
        for symbol in self.special_symbols:
            if super(WordDictionary, self).get(symbol) is None: # might be 0
                 self[symbol] = len(self)
        
        # save the indices of the special symbols
        if self.variant:
            self.padding_left = super(WordDictionary, self).get(padding_left)
            self.padding_right = super(WordDictionary, self).get(padding_right)
            self.rare = super(WordDictionary, self).get(rare)
        else:
            # there is no corresponding string in dictionary
            self.padding_left = INT(len(self))
            self.padding_right = self.padding_left
            self.rare = INT(self.padding_left + 1)
    
    def size(self):
        """
        :return: the number of words in the dictionary, excluding special symbols.
        """
        return len(self) - len(self.special_symbols)

    def save(self, file):
        """
        Saves the word dictionary to the given file as a list of words.
        Special words (paddings and rare) are also included.
        """
        pickle.dump(self.variant, file)
        pickle.dump(self.words, file)
        pickle.dump((self.rare, 
                     self.padding_left,
                     self.padding_right), file)

    @classmethod
    def load(cls, file):
        o = WordDictionary.__new__(cls)
        o.variant = pickle.load(file)
        o.words = pickle.load(file)
        o.rare, o.padding_left, o.padding_right = pickle.load(file)
        for i,x in enumerate(o.words):
            # FIXME: this assumes normalized words in file
            super(WordDictionary, o).__setitem__(x, INT(i))
        return o

    def _get_frequency_count(self, sentences):
        """
        Returns a token counter for normalized tokens in :param sentences:.
        
        :param sentences: an iterable on lists of tokens.
        """
        return Counter(self.normalize(t) for sent in sentences for t in sent)
    
    def update_tokens(self, tokens, size=None, minimum_occurrences=1, freqs=None):
        """
        Updates the dictionary, adding tokens until :param size: is reached.
        
        :param freqs: a dictionary providing a token count.
        """
        if freqs is None:
            freqs = self._get_frequency_count([tokens])
            
        if size is None or size == 0:
            # size None or 0 means no size limit
            size = len(freqs)
        
        increment = size - self.size()
        if increment <= 0:
            return
        
        # tokens not present in the dictionary and above minimum frequency 
        new_tokens = [token for token in freqs 
                      if token not in self and freqs[token] >= minimum_occurrences]
        # order the words from the most frequent to the least
        new_tokens.sort(key=lambda x: freqs[x], reverse=True)
        
        for token in new_tokens:
            self[token] = len(self)
            increment -= 1
            if increment == 0:
                break
        
    def normalize(self, word):
        """
        Normalize word, converting digits to 0 and lowercasing (when variant is 'senna').
        """
        if self.variant == 'senna':
            # senna converts numbers to '0'
            if isNumber(word):
                word = '0'
            else:
                word = word.lower()
            return re.sub('[0-9]', '0', word)
        if self.variant:
            # replace all digits by '0'
            return re.sub('[0-9]', '0', word)
        return word

    def __contains__(self, key):
        """
        Overrides the "in" operator. Case insensitive when variant is 'senna'.
        """
        # deal with symbols in original case, e.g. PADDING, UNKNOWN.
        return super(WordDictionary, self).__contains__(key) or \
            super(WordDictionary, self).__contains__(self.normalize(key))

    def __getitem__(self, key):
        """
        Overrides the [] read operator. 
        
        Two differences from the original:
        1) if the key is not present, it returns the value for the UNKNOWN key.
        2) match is attempted also with normalized key.
        """
        # deal with symbols in original case, e.g. PADDING, UNKNOWN.
        return super(WordDictionary, self).get(key) or \
            super(WordDictionary, self).get(self.normalize(key), self.rare)

    def get(self, key):
        """
        Overrides the dictionary get method, so when given a word without an entry,
        it returns the value for the UNKNOWN key.
        Note that it is NOT possible to supply a default value as in the dict class.
        """
        return self.__getitem__(key)

    def __setitem__(self, key, value):
        """
        Replaces the [] write operator.
        We store INT values.

        Words are normalized before insertion.
        """
        # deal with symbols in original case, e.g. PADDING, UNKNOWN.
        if not super(WordDictionary, self).__contains__(key):
            key = self.normalize(key)
            if not super(WordDictionary, self).__contains__(key):
                self.words.append(key)
        super(WordDictionary, self).__setitem__(key, INT(value))
        
    def add(self, word):
        if word not in self:    # invokes __contains__()
            self[word] = len(self)
        
    def get_words(self, indices):
        """
        Returns the words represented by a sequence of indices.
        Notice that this might not return the original sentence,
        since the index is not injective: two words might have the same index
        e.g. numbers '11' and '22' are mapped to '00'

        """
        return (self.words[i] if i < len(self.words) else '<UNKN>' for i in indices)
    
    def get_indices(self, words):
        """
        Returns the indices corresponding to a sequence of tokens.
        """
        return (self[w] for w in words)

class NgramDictionary(WordDictionary):
    """
    Class to store ngrams and their corresponding indices in
    the network lookup table.
    """
    def __init__(self, ngrams, size=None, minimum_occurrences=None, variant=None):
        """
        Fills a dictionary (to be used for indexing) with the most
        common ngrams.
        
        :param ngrams: a list of lists of ngrams
        :param size: Maximum number of ngram indices 
            (not including paddings, rare, etc.).
        :param minimum_occurrences: The minimum number of occurrences an ngram must 
            have in order to be included.
        :param variant: either 'polyglot' or 'senna' conventions, i.e. keep upper case, use different padding tokens.
        """
        super(NgramDictionary, self).__init__(self, ngrams, size, minimum_occurrences,
                                              variant=variant)
