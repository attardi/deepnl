#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run a POS or SRL tagger on the input data and print the results
to stdout.
"""

from __future__ import print_function
import argparse
import logging
import ipdb

# Attardi: allow executing from anywhere without installing package
import sys
import os
srcdir = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(srcdir + 'build/lib.linux-x86_64-2.7')

import nlpnet
import nlpnet.utils as utils
from nlpnet.taggers import Tagger
from nlpnet.network import Network
from nlpnet.pos.pos_reader import POSReader
from nlpnet.attributes import Suffix
from nlpnet.metadata import Metadata
from nlpnet.word_dictionary import WordDictionary
import numpy as np

senna_dump = "pos.dump"

def load_features(file):
    row, col = file.readline().split()
    row = int(row)
    col = int(col)
    a = np.ndarray((row, col))
    words = [None] * row
    for i in range(row):
        values = file.readline().split()
        words[i] = values[0]
        for j, val in enumerate(values[1:]):
            a[i,j] = float(val)
    return (words, a)

def load_weights(file):
    row, col = file.readline().split()
    row = int(row)
    col = int(col)
    a = np.ndarray((row, col))
    for i in range(row):
        line = file.readline()
        for j, val in enumerate(line.split()):
            a[i,j] = float(val)
    return a

def load_bias(file):
    col = file.readline()
    col = int(col)
    a = np.zeros(col)
    line = file.readline()
    for j, val in enumerate(line.split()):
        a[j] = float(val)
    return a

def load_network():
    """
    Loads the network from the default file and returns it.
    """
    file = open(senna_dump)
    words, type_features = load_features(file)
    word_dict = WordDictionary(None, wordlist=words, variant='senna')
    tables = [type_features]
    
    # PADDING, allcaps, hascap, initcap, nocaps
    caps, caps_features = load_features(file)
    tables.append(caps_features)

    suff, suffix_features = load_features(file)
    tables.append(suffix_features)

    hidden_weights = load_weights(file) # (hidden_size, input_size)
    hidden_bias = load_bias(file)
    output_weights = load_weights(file) # (output_size, hidden_size)
    output_bias = load_bias(file)
        
    transition0 = load_bias(file)
    transitions = load_weights(file).T
    transitions = np.vstack((transitions, transition0))

    word_window_size = 5
    input_size = hidden_weights.shape[1]
    hidden_size = hidden_weights.shape[0]
    output_size = output_bias.shape[0]
        
    nn = Network(word_window_size, input_size, hidden_size, output_size,
                 hidden_weights, hidden_bias, output_weights, output_bias)
    nn.feature_tables = tables
    nn.transitions = transitions 
    
    return nn, word_dict, suff

class SennaPOSTagger(Tagger):
    """A POSTagger loads the models and performs POS tagging on text."""
    
    def _load_data(self):
        """Loads data for POS from SENNA dump"""
        md = Metadata.load_from_file('pos')
        self.nn, word_dict, suff = load_network()
        self.reader = POSReader()
        self.reader.word_dict = word_dict
        self.reader.create_converter(md)
        self.itd = self.reader.get_inverse_tag_dictionary()
        self.nn.padding_left = self.reader.converter.get_padding_left()
        self.nn.padding_right = self.reader.converter.get_padding_right()
        self.nn.pre_padding = np.array([self.nn.padding_left] * 2)
        self.nn.pos_padding = np.array([self.nn.padding_right] * 2)
        Suffix.codes = {}
        for i, s in enumerate(suff):
            Suffix.codes[s] = i
        Suffix.other = Suffix.codes['NOSUFFIX']
    
    def tag(self, text=None):
        """
        Tags the given text.
        
        :param text: a string or unicode object. Strings assumed to be utf-8
        :returns: a list of lists (sentences with tokens).
            Each sentence has (token, tag) tuples.
        """
        result = []
        if text:
            tokens = utils.tokenize(text, clean=False)
            for sent in tokens:
                tags = self.tag_tokens(sent)
                result.append(zip(sent, tags))
        else:
            # read tsv from stdin
            sent = []
            for line in sys.stdin:
                line = line.decode('utf-8').strip()
                if line:
                    sent.append(line.split()[0])
                else:
                    #ipdb.set_trace()
                    tags = self.tag_tokens(sent)
                    result.append(zip(sent, tags))
                    sent = []

        return result
    
    def tag_tokens(self, tokens):
        """
        Tags a given list of tokens. 
        
        Tokens should be produced with the nlpnet tokenizer in order to 
        match the entries in the vocabulary. If you have non-tokenized text,
        use POSTagger.tag(text).
        
        :param tokens: a list of strings
        :returns: a list of strings (the tags)
        """
        converter = self.reader.converter
        # do not use clean_text. Attardi
        #converted_tokens = np.array([converter.convert(utils.clean_text(token, False)) 
        converted_tokens = converter.convert(tokens)
        answer = self.nn.tag_sentence(converted_tokens)
        tags = [self.itd[tag] for tag in answer]
        return tags

def process_input(task):
    """
    This function reads input from stdin and processes sentences.
    
    :param task: either 'pos' or 'ner'
    """
    task_lower = task.lower()
    if task_lower == 'pos':
        tagger = SennaPOSTagger()
    elif task_lower == 'ner':
        tagger = nlpnet.taggers.NERTagger()
    else:
        raise ValueError('Unknown task: %s' % task)
    
    result = tagger.tag()        
    _print_tagged(result, task)

def _print_tagged(tagged_sents, task):
    """
    Prints the tagged text to stdout.
    
    :param tagged_sents: sentences tagged according to any of nlpnet taggers.
    :param task: the tagging task (either 'pos' or 'ner')
    """
    if task == 'pos':
        _print_tagged_pos(tagged_sents)
    elif task == 'ner':
        _print_tagged_ner(tagged_sents)
    else:
        raise ValueError('Unknown task: %s' % task)
    
def _print_tagged_pos(tagged_sents):
    """Prints one sentence per line as token_tag"""
    # for sent in tagged_sents:
    #     s = ' '.join('_'.join(item) for item in sent)
    #     print(s)

    # print in tsv
    for sent in tagged_sents:
        for token in sent:
            print('\t'.join([item.encode('utf-8') for item in token]))
        print()

def _print_tagged_ner(tagged_sents):
    """Prints one token per line as token\ttag"""
    for sent in tagged_sents:
        for tok, tag in sent:
            print(tok[0] + '\t' + tag # tok is (form, POS))
        print()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task for which the network should be used.', 
                        type=str, choices=['pos', 'ner'])
    parser.add_argument('data', help='Directory containing trained models.', type=str)
    parser.add_argument('-v', help='Verbose mode', action='store_true', dest='verbose')
    args = parser.parse_args()
    
    nlpnet.set_data_dir(args.data)
    
    #interactive_running(args.task)
    process_input(args.task)
    
