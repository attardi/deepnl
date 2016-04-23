#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and use a convolutional neural network classifier.

Author: Giuseppe Attardi
"""

from __future__ import print_function
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
#sys.path.append(libdir)
sys.path.insert(0,libdir)

# local
from deepnl.corpus import *
from deepnl.extractors import *
from deepnl.networkconv import ConvolutionalNetwork
from deepnl.trainerconv import ConvTrainer
from deepnl.reader import ClassifyReader
from deepnl.classifier import Classifier

# ----------------------------------------------------------------------
# Auxiliary functions

def create_trainer(args, converter, labels):
    """
    Creates or loads a neural network according to the specified args.
    :param labels: list of labels.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = ConvTrainer.load(args.load)
        # change learning rate
        trainer.learning_rate = args.learning_rate
        trainer.threads = args.threads
    else:
        logger.info('Creating new network...')
        # sum the number of features in all extractors' tables 
        feat_size = converter.size()
        pool_size = args.window * 2 + 1
        nn = ConvolutionalNetwork(feat_size * pool_size, args.hidden,
                                  args.hidden2, len(labels), pool_size)
        options = {
            'learning_rate': args.learning_rate,
            'eps': args.eps,
            'verbose': args.verbose,
            'left_context': args.window,
            'right_context': args.window
        }
        trainer = ConvTrainer(nn, converter, labels, options)

    trainer.saver = saver(args.model, args.vectors, args.variant)

    logger.info("... with the following parameters:")
    logger.info(trainer.nn.description())
    
    return trainer

def saver(model_file, vectors_file, variant):
    """Function for saving model periodically"""
    def save(trainer):
        # save embeddings also separately
        if vectors_file:
            trainer.save_vectors(vectors_file, variant)
        if model_file:
            with open(model_file, 'wb') as file:
                trainer.classifier.save(file)
    return save

# ----------------------------------------------------------------------

def main():

    # set the seed for replicability
    np.random.seed(42)          # DEBUG

    defaults = {}
    
    parser = argparse.ArgumentParser(description="Convolutional network classifier.")
    
    parser.add_argument('-c', '--config', dest='config_file',
                        help='Specify config file', metavar='FILE')

    # args, remaining_argv = parser.parse_known_args()

    # if args.config_file:
    #     config = ConfigParser.SafeConfigParser()
    #     config.read([args.config_file])
    #     defaults = dict(config.items('Defaults'))

    # parser.set_defaults(**defaults)

    parser.add_argument('model', type=str,
                        help='Model file to train/use.')

    # input format
    format = parser.add_argument_group('Format')

    format.add_argument('--label-field', type=int, default=1,
                        help='Field containing label (default %(default)s).')
    format.add_argument('--text-field', type=int, default=2,
                        help='Field containing text (default %(default)s).')

    # training options
    train = parser.add_argument_group('Train')

    train.add_argument('-t', '--train', type=str, default=None,
                       help='File with annotated data for training.')

    train.add_argument('-w', '--window', type=int, default=5,
                       help='Size of the word window (default %(default)s)')
    train.add_argument('-s', '--embeddings-size', type=int, default=50,
                       help='Number of features per word (default %(default)s)',
                       dest='embeddings_size')
    train.add_argument('-e', '--epochs', type=int, default=100,
                       help='Number of training epochs (default %(default)s)',
                       dest='iterations')
    train.add_argument('-l', '--learning_rate', type=float, default=0.001,
                       help='Learning rate for network weights (default %(default)s)',
                       dest='learning_rate')
    train.add_argument('--eps', type=float, default=1e-6,
                        help='Epsilon value for AdaGrad (default %(default)s)')
    train.add_argument('-n', '--hidden', type=int, default=200,
                       help='Number of hidden neurons (default %(default)s)')
    train.add_argument('-n2', '--hidden2', type=int, default=200,
                       help='Number of hidden neurons (default %(default)s)')

    # Extractors:
    extractors = parser.add_argument_group('Extractors')
    extractors.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                            help='Include capitalization features. Optionally, supply the number of features (default %(default)s)')
    extractors.add_argument('--suffix', const=5, nargs='?', type=int, default=None,
                            help='Include suffix features. Optionally, supply the number of features (default %(default)s)')
    extractors.add_argument('--suffixes', type=str, default='',
                        help='Load suffixes from this file')
    extractors.add_argument('--prefix', const=0, nargs='?', type=int, default=None,
                        help='Include prefix features. Optionally, '\
                            'supply the number of features (default %(default)s)')
    extractors.add_argument('--prefixes', type=str, default='',
                        help='Load prefixes from this file')
    # Embeddings
    embeddings = parser.add_argument_group('Embeddings')
    embeddings.add_argument('--vocab', type=str, default=None,
                        help='Vocabulary file, either read or created')
    embeddings.add_argument('--vectors', type=str, default=None,
                        help='Embeddings file, either read or created')
    embeddings.add_argument('--min-occurr', type=int, default=3,
                        help='Minimum occurrences for inclusion in vocabulary',
                        dest='minOccurr')
    embeddings.add_argument('--load', type=str, default=None,
                        help='Load previously saved model')
    embeddings.add_argument('--variant', type=str, default=None,
                        help='Either "senna" (default), "polyglot" or "word2vec".')

    # common
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads (default %(default)s)')
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
        reader = ClassifyReader(text_field=args.text_field, label_field=args.label_field)
        # a generator (can be iterated several times)
        sentences = reader.read(args.train)

        if args.vocab and os.path.exists(args.vocab):
            if args.vectors and os.path.exists(args.vectors):
                # use supplied embeddings
                embeddings = Embeddings(vectors=args.vectors, vocab_file=args.vocab,
                                        variant=args.variant)
            else:
                # create random embeddings
                embeddings = Embeddings(args.embeddings_size, vocab_file=args.vocab,
                                        variant=args.variant)
            # collect words from the corpus
            # build vocabulary
            vocab, bigrams, trigrams = reader.create_vocabulary(sentences,
                                                                #size=args.vocab_size,
                                                                min_occurrences=args.minOccurr)
            # add them to the given vocabulary
            embeddings.merge(vocab)
            logger.info("Overriding vocabulary in %s" % args.vocab)
            embeddings.save_vocabulary(args.vocab)

        elif args.variant == 'word2vec':
            if os.path.exists(args.vectors):
                embeddings = Embeddings(vectors=args.vectors,
                                        variant=args.variant)
                vocab, bigrams, trigrams = reader.create_vocabulary(sentences,
                                                                    #args.vocab_size,
                                                                    min_occurrences=args.minOccurr)
                embeddings.merge(vocab)
            else:
                vocab, bigrams, trigrams = reader.create_vocabulary(sentences,
                                                                    #args.vocab_size,
                                                                    min_occurrences=args.minOccurr)
                embeddings = Embeddings(vocab=vocab,
                                        variant=args.variant)
            if args.vocab:
                logger.info("Saving vocabulary in %s" % args.vocab)
                embeddings.save_vocabulary(args.vocab)

        elif not args.vocab_size:
            logger.error("Missing parameter --vocab-size")
            return
        else:
            # build vocabulary and tag set
            vocab, bigrams, trigrams = reader.create_vocabulary(sentences,
                                                                #args.vocab_size,
                                                                min_occurrences=args.minOccurr)
            logger.info("Creating word embeddings")
            embeddings = Embeddings(args.embeddings_size, vocab=vocab,
                                    variant=args.variant)
            if args.vocab:
                logger.info("Saving vocabulary in %s" % args.vocab)
                embeddings.save_vocabulary(args.vocab)

        converter = Converter()
        converter.add(embeddings)

        if args.caps:
            logger.info("Creating capitalization features...")
            converter.add(CapsExtractor(args.caps))

        if ((args.suffixes and not os.path.exists(args.suffixes)) or
            (args.prefixes and not os.path.exists(args.prefixes))):
            # collect the forms once
            words = (tok for sent in sentences for tok in sent)

        if args.suffix:
            if os.path.exists(args.suffixes):
                logger.info("Loading suffix list...")
                extractor = SuffixExtractor(args.suffix, args.suffixes)
                converter.add(extractor)
            else:
                logger.info("Creating suffix list...")
                extractor = SuffixExtractor(args.suffix, None, words)
                converter.add(extractor)
                if args.suffixes:
                    logger.info("Saving suffix list to: %s", args.suffixes)
                    extractor.write(args.suffixes)

        if args.prefix:
            if os.path.exists(args.prefixes):
                logger.info("Loading prefix list...")
                extractor = PrefixExtractor(args.prefix, args.prefixes)
                converter.add(extractor)
            else:
                logger.info("Creating prefix list...")
                extractor = PrefixExtractor(args.prefix, None, words)
                converter.add(extractor)
                if args.prefixes:
                    logger.info("Saving prefix list to: %s", args.prefixes)
                    extractor.write(args.prefixes)

        # labels from all examples
        examples = [converter.convert(example) for example in sentences]
        # assign index to labels
        sent_labels = reader.polarities
        labels_index = {}
        labels = []
        for i,c in enumerate(set(sent_labels)):
            labels_index[c] = i
            labels.append(c)
        trainer = create_trainer(args, converter, labels)
        logger.info("Starting training with %d examples" % len(examples))

        report_frequency = max(args.iterations / 200, 1)
        report_frequency = 1    # DEBUG
        labels_ids = [labels_index[label] for label in sent_labels]
        trainer.train(examples, labels_ids, args.iterations, report_frequency,
                      args.threads)
    
        logger.info("Saving trained model ...")
        trainer.saver(trainer)
        logger.info("... to %s" % args.model)

    else:
        # predict
        with open(args.model) as file:
            classifier = Classifier.load(file)
        reader = ClassifyReader(text_field=args.text_field, label_field=args.label_field)
        
        for example in reader:
            words = example[reader.text_field].split()
            example[reader.label_field] = classifier.predict(words)
            print('\t'.join(example).encode('utf-8'))

# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
