# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict
import cPickle as pickle
import re
import sys                      # DEBUG

num = re.compile('[+\-]?([0-9][,.]?)+$')

def isNumber(key):
    return num.match(key)


class WordDictionary(dict):
    """
    Class to store words and their corresponding indices in
    the network lookup table. Also deals with padding and
    maps rare words to a special index.
    """
    
    # SENNA convenctions
    padding_left = u'PADDING'
    padding_right = u'PADDING'
    rare = u'UNKNOWN'
    
    def __init__(self, iterable_sent, size=None, minimum_occurrences=None, wordlist=None, variant=None):
        """
        Fills a dictionary (to be used for indexing) with the most
        common words in the given text.
        
        :param iterable_sent: an iterable list of lists of tokens 
            (each token represented as a string).
        :param size: Maximum number of token indices 
            (not including paddings, rare, etc.).
        :param minimum_occurrences: The minimum number of occurrences a token must 
            have in order to be included.
        :param wordlist: Use this list of words to build the dictionary.
            Overrides iterable_sent if not None and ignores maximum size.
        :param variant: either 'polyglot', 'word2vec', or 'senna' conventions,
            i.e. keep case, use different padding tokens.
        """
        self.variant = variant
        if variant:
            self.variant = variant.lower()
        if self.variant == 'polyglot':
            WordDictionary.padding_left = u'<PAD>'
            WordDictionary.padding_right = u'<PAD>'
            WordDictionary.rare = u'<UNK>'
        elif self.variant == 'word2vec':
            WordDictionary.padding_left = u'</s>'
            WordDictionary.padding_right = u'</s>'
            WordDictionary.rare = u'<UNK>'

        if wordlist is None:
            # work with the supplied iterable_sent. extract frequencies.
            
            # gets frequency count
            c = self._get_frequency_count(iterable_sent)
        
            if minimum_occurrences is None:
                minimum_occurrences = 1
            
            words = [key for key, number in c.most_common() 
                     if number >= minimum_occurrences]
            
            if size is not None and size < len(words):
                words = words[:size]
        
        else:
            # Keep the order and eliminate duplicates
            words = list(OrderedDict.fromkeys(self.normalize(w) for w in wordlist))
            
        # trim to the maximum size
        if size is None:
            size = len(words)
        else:
            size = min(size, len(words))
            words = words[:size]
        
        # build the inverse index
        self.words = [0] * len(words) # inverse index
        for num, word in enumerate(words):
            # self[word] = num (since words have been normalized)
            super(WordDictionary, self).__setitem__(word, num)
            self.words[num] = word
        
        # if the given words include one of the the rare or padding symbols,
        # don't replace it
        special_symbols = (WordDictionary.rare, 
                           WordDictionary.padding_left,
                           WordDictionary.padding_right)
        
        for symbol in special_symbols:
            if super(WordDictionary, self).get(symbol) is None: # might be 0
                 self[symbol] = len(self)
        
        self.check()
    
    def size(self):
        """
        :return: the number of words ins the dictionary, excluding special symbols.
        """
        l = len(self)
        special_symbols = set((WordDictionary.rare, 
                               WordDictionary.padding_left,
                               WordDictionary.padding_right))
        for symbol in special_symbols:
            l -= super(WordDictionary, self).get(symbol) is not None
        return l

    def save(self, file):
        """
        Saves the word dictionary to the given file as a list of words.
        Special words (paddings and rare) are also included.
        """
        pickle.dump(self.variant, file)
        pickle.dump(self.words, file)
        # FIXME: these are shared among all extractors
        pickle.dump((WordDictionary.rare, 
                     WordDictionary.padding_left,
                     WordDictionary.padding_right,
                     self.index_rare), file)

    @classmethod
    def load(cls, file):
        o = WordDictionary.__new__(cls)
        o.variant = pickle.load(file)
        o.words = pickle.load(file)
        (WordDictionary.rare, WordDictionary.padding_left,  WordDictionary.padding_right, o.index_rare) = pickle.load(file)
        for i,x in enumerate(o.words):
            #o[x] = i
            super(WordDictionary, o).__setitem__(x, i)
        return o

    def _get_frequency_count(self, iterable_sent):
        """
        Returns a token counter for normalized tokens in :param iterable_sent:.
        
        :param iterable_sent: an iterable list of lists of tokens.
        """
        return Counter(self.normalize(t) for sent in iterable_sent for t in sent)
    
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
        
        if self.num_tokens >= size:
            return

        increment = size - self.num_tokens
        
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
        
        self.check()
    
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
        # replace all digits by '0'
        return re.sub('[0-9]', '0', word)

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
        1) when given a word without an entry, it returns the value for the
           UNKNOWN key.
        2) entries are converted, replacing digits with 0 (and lower cased
           when variant is 'senna').
        """
        # deal with symbols in original case, e.g. PADDING, UNKNOWN.
        idx = super(WordDictionary, self).get(key)
        if idx is not None:     # might be 0
            return idx
        return super(WordDictionary, self).get(self.normalize(key), self.index_rare)

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

        Words are normalized before insertion.
        """
        # deal with symbols in original case, e.g. PADDING, UNKNOWN.
        idx = super(WordDictionary, self).get(key, -1)
        if idx == -1:
            key = self.normalize(key)
            self.words.append(key)
        super(WordDictionary, self).__setitem__(key, value)
        
    def add(self, word):
        if word not in self:
            self[word] = len(self)

    def check(self):
        """
        Checks the internal structure of the dictionary and makes necessary
        adjustments, such as updating num_tokens.
        """

        # must repeat, since it is called in reader.load_dictionary()
        # after reloading from dump.
        # FIXME: still needed?
        if self.variant == 'polyglot':
            WordDictionary.padding_left = u'<PAD>'
            WordDictionary.padding_right = u'<PAD>'
            WordDictionary.rare = u'<UNK>'
        elif self.variant == 'word2vec':
            WordDictionary.padding_left = u'</s>'
            WordDictionary.padding_right = u'</s>'
            WordDictionary.rare = u'<UNK>'

        # Keep case for special tokens.
        self.index_padding_left = super(WordDictionary, self).get(WordDictionary.padding_left)
        self.index_padding_right = super(WordDictionary, self).get(WordDictionary.padding_right)
        self.index_rare = super(WordDictionary, self).get(WordDictionary.rare)
        # Polyglot and Senna have two special tokens
        if self.index_padding_left == self.index_padding_right:
            self.num_tokens = len(self) - 2
        else:
            self.num_tokens = len(self) - 3
        
    def get_words(self, indices):
        """
        Returns the words represented by a sequence of indices.
        Notice that this might not return the original sentence,
        since the index is not injective: two words might have the same index
        e.g. numbers '11' and '22' are mapped to '00'

        """
        words = [self.words[i] if i < len(self.words) else '<UNKN>' for i in indices]
        return words
    
    def get_indices(self, words):
        """
        Returns the indices corresponding to a sequence of tokens.
        """
        indices = [self[w] for w in words]
        return indices

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
        WordDictionary.__init__(self, ngrams, size, minimum_occurrences,
                                 variant=variant)
