#!/usr/bin/env python

"""
Convert embeddings to nlpnet format.

Usage:
   embeddingsAdapter.py [Options] types-features-pos.npy word-dict.pickle

Output:
   types-features-pos.npy  : embeddings file
   word-dict.pickle        : word dictionary file

Options:
  -h, --help               : display this help and exit
  -p, --polyglot file      : polyglot file
  -s, --senna file         : senna file
  -w, --word2embedding file: word2embedding file
  -v, --vocabulary file    : word2embedding vocabulary file
"""

import sys, os, getopt, pickle, numpy

sys.path.append('.')
sys.path.append('../../embeddings/word2embeddings')
from nlpnet.word_dictionary import WordDictionary

def convertSenna(sennaFile, types_feats_file, word_dict_file):
    words = []
    embeddings = []
    for line in open(sennaFile):
        items = line.split()
        words.append(items[0].decode('utf-8'))
        embeddings.append([float(x) for x in items[1:]])
    numpy.save(types_feats_file, embeddings)
    out = open(word_dict_file, 'w')
    pickle.dump(WordDictionary(None, wordlist=words, variant='senna'), out)

def convertPolyglot(polyglotFile, types_feats_file, word_dict_file):
    words, embeddings = pickle.load(open(polyglotFile))
    numpy.save(types_feats_file, embeddings)
    out = open(word_dict_file, 'w')
    pickle.dump(WordDictionary(None, wordlist=words, variant='polyglot'), out)

def convertWord2embedding(word2embeddingsFile, types_feats_file, word_dict_file, vocabularyFile):
    m = pickle.load(open(word2embeddingsFile))
    numpy.save(types_feats_file, m.get_word_embeddings())

    words = ['<UNK>', '<S>', '</S>', '<PAD>']
    words.extend(open(vocabularyFile, 'rb').read().decode('utf-8').strip().splitlines())
    out = open(word_dict_file, 'w')
    pickle.dump(WordDictionary(None, wordlist=words, variant='polyglot'), out)

def showHelp():
    print >> sys.stderr, __doc__,

def main():
    try:
        longOpts = ['help', 'polyglot', 'senna', 'word2embedding', 'vocabulary']
        opts, args = getopt.getopt(sys.argv[1:], 'hp:s:w:v:', longOpts)
    except getopt.GetoptError:
        showHelp()
        sys.exit(1)

    polyglotFile = None
    sennaFile = None
    word2embeddingFile = None
    vocabulary = None

    for opt, arg in opts:
        if opt in ['-h', '--help']:
            showHelp()
            sys.exit()
        elif opt in ['-p', '--polyglot']:
            polyglotFile = arg
        elif opt in ['-s', '--senna']:
            sennaFile = arg
        elif opt in ['-w', '--word2embedding']:
            word2embeddingFile = arg
        elif opt in ['-v', '--vocabulary']:
            vocabulary = arg

    if len(args) != 2:
        showHelp()
        sys.exit(1)

    embeddingsfile = args[0]
    word_dict_file = args[1]

    try:
        os.makedirs(os.path.dirname(word_dict_file))
        os.makedirs(os.path.dirname(types_feats_file))
    except:
        pass

    if polyglotFile:
        convertPolyglot(polyglotFile, embeddingsfile, word_dict_file)
    elif sennaFile:
        convertSenna(sennaFile, embeddingsfile, word_dict_file)
    elif word2embeddingFile:
        convertWord2embedding(word2embeddingFile, embeddingsfile, word_dict_file, vocabulary)
    else:
        showHelp()

if __name__ == '__main__':
    main()
