#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and use a convolutional neural network.

Author: Giuseppe Attardi
"""

import logging
import numpy as np
import argparse
from ConfigParser import ConfigParser

# allow executing from anywhere without installing the package
import sys
import os
import distutils.util
builddir = os.path.dirname(os.path.realpath(__file__)) + '/../build/lib.'
libdir = builddir + distutils.util.get_platform() + '-' + '.'.join(map(str, sys.version_info[:2]))
sys.path.append(libdir)

# local
from deepnl.corpus import *
from deepnl.extractors import *
from deepnl.networkconv import ConvolutionalNetwork
from deepnl.trainerconv import ConvTrainer

# ----------------------------------------------------------------------
# Auxiliary functions

def create_trainer(args, converter, labels):
    """
    Creates or loads a neural network according to the specified args.
    :param labels: dict of labels.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = ConvTrainer.load(args.load)
        trainer.learning_rate = args.learning_rate
        trainer.threads = args.threads
    else:
        logger.info('Creating new network...')
        trainer = ConvTrainer(converter, args.learning_rate,
                              args.window/2, args.window/2,
                              args.hidden, labels, args.verbose)

    trainer.saver = saver(args.model, args.output)

    logger.info("... with the following parameters:")
    logger.info(trainer.nn.description())
    
    return trainer

def saver(model_file, vectors_file):
    """Function for saving model periodically"""
    def save(trainer):
        # save embeddings also separately
        if vectors_file:
            trainer.converter.extractors[0].save_vectors(vectors_file)
        with open(model_file, 'wb') as file:
            trainer.tagger.save(file)
    return save

# ----------------------------------------------------------------------

def main():

    # set the seed for replicability
    np.random.seed(42)          # DEBUG

    defaults = {}
    
    parser = argparse.ArgumentParser(description="Train or use neural network.")
    
    parser.add_argument('-c', '--config', dest='config_file',
                        help='Specify config file', metavar='FILE')

    #args, remaining_argv = parser.parse_known_args()

    # if args.config_file:
    #     config = ConfigParser.SafeConfigParser()
    #     config.read([args.config_file])
    #     defaults = dict(config.items('Defaults'))

    # parser.set_defaults(**defaults)

    parser.add_argument('model', type=str,
                        help='Model file to train/use.')
    parser.add_argument('-w', '--window', type=int, default=5,
                        help='Size of the word window (default 5)')
    parser.add_argument('-s', '--embeddings-size', type=int, default=50,
                        help='Number of features per word (default 50)',
                        dest='embeddings_size')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs (default 100)',
                        dest='iterations')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
                        help='Learning rate for network weights (default 0.001)',
                        dest='learning_rate')
    parser.add_argument('-n', '--hidden', type=int, default=200,
                        help='Number of hidden neurons (default 200)',
                        dest='hidden')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads (default 1)')
    parser.add_argument('-t', '--train', type=str, default=None,
                        help='File with annotated data for training.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='File where to save embeddings')

    # Extractors:
    parser.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                        help='Include capitalization features. Optionally, supply the number of features (default 5)')
    parser.add_argument('--suffix', const=5, nargs='?', type=int, default=None,
                            help='Include suffix features. Optionally, supply the number of features (default 5)')
    parser.add_argument('--suffixes', type=str,
                        help='Load suffixes from this file')
    parser.add_argument('--prefix', const=0, nargs='?', type=int, default=None,
                        help='Include prefix features. Optionally, '\
                        'supply the number of features (default 0)')
    parser.add_argument('--prefixes', type=str,
                        help='Load prefixes from this file')
    # common
    parser.add_argument('--vocab', type=str, default=None,
                        help='Vocabulary file, either read or created')
    parser.add_argument('--vectors', type=str, default=None,
                        help='Embeddings file, either read or created')
    parser.add_argument('--min-occurr', type=int, default=3,
                        help='Minimum occurrences for inclusion in vocabulary',
                        dest='minOccurr')
    parser.add_argument('--load', type=str, default=None,
                        help='Load previously saved model')
    parser.add_argument('--variant', type=str, default=None,
                        help='Either "senna" (default), "polyglot" or "word2vec".')
    parser.add_argument('-v', '--verbose', help='Verbose mode',
                        action='store_true')

    # Use this for obtaining defaults from config file:
    #args = arguments.get_args()
    args = parser.parse_args()

    log_format = '%(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format=log_format, level=log_level)
    logger = logging.getLogger("Logger")

    config = ConfigParser()
    if args.config_file:
        config.read(args.config_file)

    # merge args with config

    if args.train:
        reader = TweetReader()
        # a generator (can be iterated several times)
        examples = reader.read(args.train)

        if args.vocab:
            if not args.vectors:
                logger.error("No --vectors specified")
                return
            embeddings = Embeddings(args.embeddings_size, args.vocab,
                                    args.vectors, variant=args.variant)
        elif args.variant == 'word2vec':
            embeddings = Embeddings(vectors=args.vectors,
                                    variant=args.variant)
        else:
            # build vocabulary and labels
            vocab = reader.create_vocabulary(examples,
                                             args.vocab_size,
                                             args.minOccurr)
            logger.info("Creating word embeddings")
            embeddings = Embeddings(args.embeddings_size, vocab=vocab,
                                    variant=args.variant)
        converter = Converter()
        converter.add(embeddings)

        if args.caps:
            logger.info("Creating capitalization features...")
            converter.add(CapsExtractor(args.caps))

        if args.suffix:
            logger.info("Creating suffix features...")
            # collect the forms
            words = (tok[0] for sent in examples for tok in sent)
            extractor = SuffixExtractor(args.suffix, args.suffixes, words)
            converter.add(extractor)

        if args.prefix:
            logger.info("Creating prefix features...")
            extractor = PrefixExtractor(args.prefix, args.prefixes, examples)
            converter.add(extractor)

        # labels from all examples
        labels = reader.polarities
        labels_dict = {c:i for i,c in enumerate(set(labels))}
        examples = []
        for example in reader.sentences:
            examples.append(converter.convert([token[0] for token in example]))

        trainer = create_trainer(args, converter, labels_dict)
        logger.info("Starting training with %d examples" % len(examples))

        report_frequency = max(args.iterations / 200, 1)
        report_frequency = 1    # DEBUG
        trainer.train(examples, labels, args.iterations, report_frequency,
                      args.threads)
    
        logger.info("Saving trained model ...")
        trainer.saver(trainer)
        logger.info("... to %s" % args.model)

    else:
        # predict
        with open(args.model) as file:
            classifier = Classifier.load(file)
        reader = TsvReader()
        for tweet in reader:
            text = tweet[-1]
            tweet[TweetReader.polarity_field] = classifier.predict(text)
            print tweet

# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
