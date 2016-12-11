#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learn word embeddings from plain text.

Author: Giuseppe Attardi
"""

import logging
import numpy as np
import argparse
from ConfigParser import ConfigParser

# profiling
# import yappi

# allow executing from anywhere without installing the package
import sys
import os
import distutils.util
builddir = os.path.dirname(os.path.realpath(__file__)) + '/../build/lib.'
libdir = builddir + distutils.util.get_platform() + '-' + '.'.join(map(str, sys.version_info[:2]))
sys.path.append(libdir)

# local
from deepnl.extractors import *
from deepnl.reader import TextReader
from deepnl.network import Network
from deepnl.words import LmTrainer

# ----------------------------------------------------------------------
# Auxiliary functions


def create_trainer(args, converter):
    """
    Creates or loads a neural network according to the specified args.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = LmTrainer.load(args.load)
        trainer.learning_rate = args.learning_rate
    else:
        logger.info('Creating new network...')
        # sum the number of features in all extractors' tables
        input_size = converter.size() * (args.windows * 2 + 1)
        nn = LmNetwork(input_size, args.hidden, 1)
        options = {
            'learning_rate': args.learning_rate,
            'eps': args.eps,
            'ro': args.ro,
            'verbose': args.verbose,
            'left_context': args.window,
            'right_context': args.window,
            'ngram_size': args.ngrams
        }
        trainer = LmTrainer(nn, converter, options)

    trainer.saver = saver(args.output, args.vectors)

    logger.info("... with the following parameters:")
    logger.info(trainer.nn.description())

    return trainer


def saver(model_file, vectors_file):
    """Function for saving model periodically"""
    def save(trainer):
        # save embeddings also separately
        if vectors_file:
            trainer.save_vectors(vectors_file)
        trainer.save(model_file)
    return save

# ----------------------------------------------------------------------


def main():

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
    parser.add_argument('--ngrams', type=int, default=1,
                        help='Size of ngrams (default %(default)s)')
    parser.add_argument('--train', type=str, default=None,
                        help='File with text corpus for training.', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='File where to save model, for further training')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Vocabulary file')
    parser.add_argument('--vectors', required=True,
                        help='Embeddings file, either read and updated or created')
    parser.add_argument('--load', type=str,
                        help='Load previously saved model')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads (default %(default)s)')
    parser.add_argument('--words', type=int, default=0,
                        help='Number of words in corpus')
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

    if not os.path.exists(args.vocab):
        logger.error("Missing vocabulary: " + args.vocab)
        return

    embeddings = Embeddings(args.embeddings_size, args.vocab, args.vectors,
                            variant=args.variant)

    logger.info("Read data")
    converter = Converter()
    converter.add(embeddings)

    trainer = create_trainer(args, converter)

    report_intervals = max(args.iterations / 200, 1)
    report_intervals = 10000    # DEBUG

    logger.info("Starting training")

    reader = TextReader()
    # a generator (can be iterated several times)
    sentences = reader.read(args.train)
    converted_sentences = converter.generator(sentences)

    trainer.train(converted_sentences, args.iterations, report_intervals,
                  args.threads, epoch_pairs=args.words)

    logger.info("Saving vectors ...")
    trainer.save_vectors(args.vectors)
    logger.info("... to %s" % args.vectors)

    if args.output:
        logger.info("Saving trained model ...")
        trainer.save(args.output)
        logger.info("... to %s" % args.output)

# ----------------------------------------------------------------------

profile = False

if __name__ == '__main__':
    # if profile:
    #     #yappi.start() # done after thread creation
    #     main()
    #     yappi.get_func_stats().print_all()
    #     yappi.get_thread_stats().print_all()
    # else:
        main()
