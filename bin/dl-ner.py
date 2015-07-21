#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and use a NE tagger.

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
#sys.path.append(libdir)      # DEBUG
sys.path.insert(0, libdir)      # DEBUG

# local
from deepnl.reader import ConllReader
from deepnl.corpus import ConllWriter
from deepnl.extractors import *
from deepnl.tagger import Tagger
from deepnl.ner_tagger import NerReader, NerTagger
from deepnl.trainer import TaggerTrainer

# ----------------------------------------------------------------------
# Auxiliary functions

def create_trainer(args, converter, tags_dict):
    """
    Creates or loads a neural network according to the specified args.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = TaggerTrainer.load(args.load)
        trainer.learning_rate = args.learning_rate
        trainer.threads = args.threads
    else:
        logger.info('Creating new network...')
        trainer = TaggerTrainer(converter, args.learning_rate,
                                args.window/2, args.window/2,
                                args.hidden, tags_dict, args.verbose)

    trainer.saver = saver(args.model, args.vectors, args.variant)

    logger.info("... with the following parameters:")
    logger.info(trainer.nn.description())
    
    return trainer

def saver(model_file, vectors_file, variant):
    """Function for saving model periodically"""
    def save(trainer):
        # save embeddings also separately
        if vectors_file:
            trainer.converter.extractors[0].dump(vectors_file, variant)
        with open(model_file, 'wb') as file:
            trainer.tagger.save(file)
    return save

# ----------------------------------------------------------------------

def main():

    # set the seed for replicability
    np.random.seed(42)

    defaults = {}
    
    parser = argparse.ArgumentParser(description="Train or use a Named Entity tagger.")
    
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
    parser.add_argument('-t', '--train', type=str, default='',
                        help='File with annotated data for training.')
    parser.add_argument('-o', '--output', type=str, default='',
                        help='File where to save embeddings')

    # Extractors:
    parser.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                        help='Include capitalization features. Optionally, supply the number of features (default 5)')
    parser.add_argument('--suffix', const=5, nargs='?', type=int, default=None,
                        help='Include suffix features. Optionally, supply the number of features (default 5)')
    parser.add_argument('--suffixes', type=str, default='',
                        help='Load suffixes from this file')
    parser.add_argument('--prefix', const=0, nargs='?', type=int, default=None,
                        help='Include prefix features. Optionally, '\
                        'supply the number of features (default 0)')
    parser.add_argument('--prefixes', type=str, default='',
                        help='Load prefixes from this file')
    parser.add_argument('--gazetteer', type=str,
                        help='Load gazetteer from this file')
    parser.add_argument('--gsize', type=int, default=5,
                        help='Size of gazetteer features (default 5)')

    # reader
    parser.add_argument('--form-field', type=int, default=0,
                        dest='formField',
                        help='Token field containin form (default 0)')
    
    # common
    parser.add_argument('--vocab', type=str, default='',
                        help='Vocabulary file, either read or created')
    parser.add_argument('--vocab-size', type=int, default=0,
                        dest='vocab_size',
                        help='Size of vocabulary to create')
    parser.add_argument('--vectors', type=str, default='',
                        help='Embeddings file, either read or created')
    parser.add_argument('--min-occurr', type=int, default=3,
                        help='Minimum occurrences for inclusion in vocabulary',
                        dest='minOccurr')
    parser.add_argument('--load', type=str, default='',
                        help='Load previously saved model')
    parser.add_argument('--variant', type=str, default='',
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
        reader = NerReader(args.formField)

        # a generator (can be iterated several times)
        sentence_iter = reader.read(args.train)

        if os.path.exists(args.vocab):
            # start with the given vocabulary
            base_vocab = reader.load_vocabulary(args.vocab)
            if os.path.exists(args.vectors):
                embeddings = Embeddings(vectors=args.vectors, vocab=base_vocab,
                                        variant=args.variant)
            else:
                embeddings = Embeddings(args.embeddings_size, vocab=base_vocab,
                                        variant=args.variant)
            # add the ngrams from the corpus
            # build vocabulary and tag set
            vocab, tagset = reader.create_vocabulary(sentence_iter,
                                                     args.vocab_size,
                                                     args.minOccurr)
            embeddings.merge(vocab)
            logger.info("Overriding vocabulary in %s" % args.vocab)
            embeddings.save_vocabulary(args.vocab)

        elif args.vocab:
            if not args.vectors:
                logger.error("No --vectors specified")
                return
            embeddings = Embeddings(args.embeddings_size, args.vocab,
                                    args.vectors, variant=args.variant)
            tagset = reader.create_tagset(sentence_iter)
            logger.info("Creating vocabulary in %s" % args.vocab)
            embeddings.save_vocabulary(args.vocab)

        elif args.variant == 'word2vec':
            if os.path.exists(args.vectors):
                embeddings = Embeddings(vectors=args.vectors,
                                        variant=args.variant)
                vocab, tagset = reader.create_vocabulary(sentence_iter,
                                                         args.vocab_size,
                                                         args.minOccurr)
                embeddings.merge(vocab)
            else:
                embeddings = Embeddings(vectors=args.vectors,
                                        variant=args.variant)
                tagset = reader.create_tagset(sentence_iter)
        else:
            # build vocabulary and tag set
            vocab, tagset = reader.create_vocabulary(sentence_iter,
                                                     args.vocab_size,
                                                     args.minOccurr)
            logger.info("Creating vocabulary in %s" % args.vocab)
            embeddings.save_vocabulary(args.vocab)
            logger.info("Creating word embeddings")
            embeddings = Embeddings(args.embeddings_size, vocab=vocab,
                                    variant=args.variant)

        converter = Converter()
        converter.add(embeddings)

        if args.caps:
            logger.info("Creating capitalization features...")
            converter.add(CapsExtractor(args.caps))

        if ((args.suffixes and not os.path.exists(args.suffixes)) or
            (args.prefixes and not os.path.exists(args.prefixes))):
            # collect the forms once
            words = (tok[reader.formField] for sent in sentence_iter for tok in sent)
        if os.path.exists(args.suffixes):
            logger.info("Loading suffix list...")
            extractor = SuffixExtractor.create(args.suffix, args.suffixes)
            converter.add(extractor)
        elif args.suffixes:
            logger.info("Creating suffix list...")
            # collect the forms
            words = (tok[reader.formField] for sent in sentence_iter for tok in sent)
            extractor = SuffixExtractor(args.suffix, None, words)
            converter.add(extractor)
            logger.info("Saving suffix list to: %s", args.suffixes)
            extractor.write(args.suffixes)

        if os.path.exists(args.prefixes):
            logger.info("Loading prefix list...")
            extractor = PrefixExtractor.create(args.prefix, args.prefixes)
            converter.add(extractor)
        elif args.prefixes:
            logger.info("Creating prefix list...")
            extractor = PrefixExtractor(args.prefix, None, words)
            converter.add(extractor)
            if args.prefixes:
                logger.info("Saving prefix list to: %s", args.prefixes)
                extractor.write(args.prefixes)

        if os.path.exists(args.gazetteer):
            logger.info("Loading gazetteers")
            for extractor in GazetteerExtractor.create(args.gazetteer, args.gsize):
                converter.add(extractor)
        elif args.gazetteer:
            logger.info("Creating gazetteer")
            tries = GazetteerExtractor.build(sentence_iter, reader.formField, reader.tagField)
            for tag, trie in tries.items():
                converter.add(GazetteerExtractor(trie, args.gsize))
            logger.info("Saving gazetter list to: %s", args.gazetteer)
            with open(args.gazetteer, 'wb') as file:
                for tag, trie in tries.iteritems():
                    for ngram in trie:
                        print >> file, ('%s\t%s' % (tag, ' '.join(ngram))).encode('UTF-8')

        # if args.pos:
        #     converter.add(POS(arg.pos))

        # obtain the tags for each sentence
        tags_dict = { t:i for i,t in enumerate(tagset) }
        sentences = []
        tags = []
        for sent in sentence_iter:
            sentences.append(converter.convert([token[reader.formField] for token in sent]))
            tags.append(np.array([tags_dict[token[reader.tagField]] for token in sent]))
    
        trainer = create_trainer(args, converter, tags_dict)
        logger.info("Starting training with %d sentences" % len(sentences))

        report_frequency = max(args.iterations / 200, 1)
        report_frequency = 1    # DEBUG
        trainer.train(sentences, tags, args.iterations, report_frequency,
                      args.threads)
    
        logger.info("Saving trained model ...")
        trainer.saver(trainer)
        logger.info("... to %s" % args.model)

    else:
        with open(args.model) as file:
            tagger = NerTagger.load(file)
        reader = ConllReader()
        for sent in reader:
            sent = [x[args.formField] for x in sent] # extract form
            ConllWriter.write(tagger.tag(sent))


# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
