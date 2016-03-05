#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learn sentiment-specific word embeddings from tweets.

Author: Giuseppe Attardi
"""

import logging
import numpy as np
import argparse
from ConfigParser import ConfigParser
from itertools import chain

# allow executing from anywhere without installing the package
import sys
import os
import distutils.util
builddir = os.path.dirname(os.path.realpath(__file__)) + '/../build/lib.'
libdir = builddir + distutils.util.get_platform() + '-' + '.'.join(map(str, sys.version_info[:2]))
#sys.path.append(libdir)
sys.path.insert(0, libdir)

# local
from deepnl import *
from deepnl.extractors import *
from deepnl.reader import TweetReader
from deepnl.network import Network
from deepnl.sentiwords import SentimentTrainer

# ----------------------------------------------------------------------
# Auxiliary functions

def create_trainer(args, converter):
    """
    Creates or loads a neural network according to the specified args.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = SentimentTrainer.load(args.load)
        # change learning rate
        trainer.learning_rate = args.learning_rate
    else:
        logger.info('Creating new network...')
        # sum the number of features in all extractors' tables 
        input_size = converter.size() * (args.window * 2 + 1)
        nn = Network(input_size, args.hidden, 2)
        options = {
            'learning_rate': args.learning_rate,
            'eps': args.eps,
            'ro': args.ro,
            'verbose': args.verbose,
            'left_context': args.window,
            'right_context': args.window,
            'ngram_size': args.ngrams,
            'alpha': args.alpha
        }
        trainer = SentimentTrainer(nn, converter, options)

    trainer.saver = saver(args.model, args.vectors)

    logger.info("... with the following parameters:")
    logger.info(trainer.nn.description())
    
    return trainer

def saver(model_file, vectors_file):
    """Function for saving model periodically"""
    def save(trainer):
        # save embeddings also separately
        if vectors_file:
            trainer.save_vectors(vectors_file)
        if model_file:
            trainer.save(model_file)
    return save

# ----------------------------------------------------------------------

if __name__ == '__main__':

    # set the seed for replicability
    np.random.seed(42)

    defaults = {}
    
    parser = argparse.ArgumentParser(description="Learn word embeddings.")
    
    parser.add_argument('-c', '--config', dest='config_file',
                        help='Specify config file', metavar='FILE')

    # args, remaining_argv = parser.parse_known_args()

    # if args.config_file:
    #     config = ConfigParser.SafeConfigParser()
    #     config.read([args.config_file])
    #     defaults = dict(config.items('Defaults'))

    # parser.set_defaults(**defaults)

    parser.add_argument('-w', '--window', type=int, default=5,
                        help='Size of the word window (default %(default)s)',
                        dest='window')
    parser.add_argument('-s', '--embeddings-size', type=int, default=50,
                        help='Number of features per word (default %(default)s)',
                        dest='embeddings_size')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs (default %(default)s)',
                        dest='iterations')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='Learning rate for network weights (default %(default)s)',
                        dest='learning_rate')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon value for AdaGrad (default %(default)s)')
    parser.add_argument('--ro', type=float, default=0.95,
                        help='Ro value for AdaDelta (default %(default)s)')
    parser.add_argument('-n', '--hidden', type=int, default=200,
                        help='Number of hidden neurons (default %(default)s)')
    parser.add_argument('--ngrams', type=int, default=2,
                        help='Length of ngrams (default %(default)s)')
    parser.add_argument('--textField', type=int, default=3,
                        help='field containing text (default %(default)s)')
    parser.add_argument('--tagField', type=int, default=2,
                        help='field containing polarity (default %(default)s)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Relative weight of normal wrt sentiment score (default %(default)s)')
    parser.add_argument('train', type=str,
                        help='File with text corpus for training.')
    parser.add_argument('--model', type=str, default=None,
                        help='File where to save the model')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Vocabulary file, either read and updated or created')
    parser.add_argument('--min-occurr', type=int, default=3,
                        help='Minimum occurrences for inclusion in vocabulary (default %(default)s',
                        dest='minOccurr')
    parser.add_argument('--vocab-size', type=int, default=0,
                        help='Maximum size of vocabulary from corpus (default %(default)s)')
    parser.add_argument('--vectors', type=str, required=True,
                        help='Embeddings file, either read and updated or created')
    parser.add_argument('--load', type=str, default=None,
                        help='Load previously saved model')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads (default %(default)s)')
    parser.add_argument('--variant', type=str, default=None,
                        help='Either "senna" (default), "polyglot" or "word2vec".')
    parser.add_argument('-v', '--verbose', help='Verbose mode',
                        action='store_true')

    args = parser.parse_args()

    log_format = '%(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format=log_format, level=log_level)
    logger = logging.getLogger("Logger")

    config = ConfigParser()
    if args.config_file:
        config.read(args.config_file)

    # merge args with config

    reader = TweetReader(text_field=args.textField, label_field=args.tagField, ngrams=args.ngrams)
    reader.read(args.train)
    vocab, bigrams, trigrams = reader.create_vocabulary(reader.sentences,
                                                        args.vocab_size,
                                                        min_occurrences=args.minOccurr)
    if args.variant == 'word2vec' and os.path.exists(args.vectors):
        embeddings = Embeddings(vectors=args.vectors, variant=args.variant)
        embeddings.merge(vocab)
        logger.info("Saving vocabulary in %s" % args.vocab)
        embeddings.save_vocabulary(args.vocab)
    elif os.path.exists(args.vocab):
        # start with the given vocabulary
        base_vocab = reader.load_vocabulary(args.vocab)
        if os.path.exists(args.vectors):
            # load embeddings
            embeddings = Embeddings(vectors=args.vectors, vocab=base_vocab,
                                    variant=args.variant)
        else:
            # create embeddings
            embeddings = Embeddings(args.embeddings_size, vocab=base_vocab,
                                    variant=args.variant)
        # add the ngrams from the corpus
        embeddings.merge(vocab)
        logger.info("Overriding vocabulary in %s" % args.vocab)
        embeddings.save_vocabulary(args.vocab)
    else:
        embeddings = Embeddings(args.embeddings_size, vocab=vocab,
                                variant=args.variant)
        logger.info("Saving vocabulary in %s" % args.vocab)
        embeddings.save_vocabulary(args.vocab)

    # Assume bigrams are prefix of trigrams, or else we should put a terminator
    # on trie
    trie = {}
    for b in chain(bigrams, trigrams):
        tmp = trie
        for w in b:
            tmp = tmp.setdefault(embeddings.dict[w], {})

    converter = Converter()
    converter.add(embeddings)

    trainer = create_trainer(args, converter)

    report_intervals = max(args.iterations / 200, 1)
    report_intervals = 10000    # DEBUG

    logger.info("Starting training")

    # a generator expression (can be iterated several times)
    # It caches converted sentences, avoiding repeated conversions
    converted_sentences = converter.generator(reader.sentences, cache=True)
    trainer.train(converted_sentences, reader.polarities, trie,
                  args.iterations, report_intervals)
    
    logger.info("Overriding vectors to %s" % args.vectors)
    embeddings.save_vectors(args.vectors, args.variant)
    if args.model:
        logger.info("Saving trained model to %s" % args.model)
        trainer.save(args.model)
