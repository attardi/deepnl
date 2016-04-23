#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and use a NE tagger.

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
#sys.path.append(libdir)      # DEBUG
sys.path.insert(0, libdir)      # DEBUG

# local
from deepnl.reader import TaggerReader
from deepnl.corpus import ConllWriter
from deepnl.extractors import *
from deepnl.ner_tagger import NerReader, NerTagger
from deepnl.trainer import TaggerTrainer
from deepnl.networkseq import SequenceNetwork

# ----------------------------------------------------------------------
# Auxiliary functions

def create_trainer(args, converter, tag_index):
    """
    Creates or loads a neural network according to the specified args.
    :param tag_index: dict of tags.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = TaggerTrainer.load(args.load)
        # change learning rate
        trainer.learning_rate = args.learning_rate
        trainer.threads = args.threads
    else:
        logger.info('Creating new network...')
        # sum the number of features in all tables 
        input_size = converter.size() * (args.window * 2 + 1)
        nn = SequenceNetwork(input_size, args.hidden, len(tag_index))
        options = {
            'learning_rate': args.learning_rate,
            'eps': args.eps,
            'ro': args.ro,
            'verbose': args.verbose,
            'left_context': args.window,
            'right_context': args.window
        }
        trainer = TaggerTrainer(nn, converter, tag_index, options)

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
        with open(model_file, 'wb') as file:
            trainer.tagger.save(file)
    return save

# ----------------------------------------------------------------------

def main():

    # set the seed for replicability
    np.random.seed(89) #(42)

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

    # training options
    train = parser.add_argument_group('Train')
    train.add_argument('-t', '--train', type=str, default='',
                        help='File with annotated data for training.')
    train.add_argument('-w', '--window', type=int, default=2,
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
    train.add_argument('-n', '--hidden', type=int, default=200,
                        help='Number of hidden neurons (default %(default)s)',
                        dest='hidden')
    train.add_argument('--eps', type=float, default=1e-6,
                        help='Epsilon value for AdaGrad (default %(default)s)')
    train.add_argument('--ro', type=float, default=0.95,
                        help='Ro value for AdaDelta (default %(default)s)')
    train.add_argument('-o', '--output', type=str, default='',
                        help='File where to save embeddings')

    # Embeddings
    embeddings = parser.add_argument_group('Embeddings')
    embeddings.add_argument('--vocab', type=str, default='',
                        help='Vocabulary file, either read or created')
    embeddings.add_argument('--vocab-size', type=int, default=0,
                            help='Maximum size of vocabulary from corpus (default %(default)s)')
    embeddings.add_argument('--vectors', type=str, default='',
                        help='Embeddings file, either read or created')
    embeddings.add_argument('--min-occurr', type=int, default=3,
                        help='Minimum occurrences for inclusion in vocabulary (default %(default)s',
                        dest='minOccurr')
    embeddings.add_argument('--load', type=str, default='',
                        help='Load previously saved model')
    embeddings.add_argument('--variant', type=str, default='',
                        help='Either "senna" (default), "polyglot" or "word2vec".')

    # Extractors:
    extractors = parser.add_argument_group('Extractors')
    extractors.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                        help='Include capitalization features. Optionally, supply the number of features (default %(default)s)')
    extractors.add_argument('--pos', const=1, type=int, nargs='?', default=None,
                        help='Use POS tag. Optionally supply the POS token field index (default %(default)s)')
    extractors.add_argument('--suffix', const=5, nargs='?', type=int, default=None,
                            help='Include suffix features. Optionally, supply the number of features (default %(default)s)')
    extractors.add_argument('--suffixes', type=str, default='',
                        help='Load suffixes from this file')
    extractors.add_argument('--prefix', const=5, nargs='?', type=int, default=None,
                            help='Include prefix features. Optionally, '\
                            'supply the number of features (default %(default)s)')
    extractors.add_argument('--prefixes', type=str, default='',
                        help='Load prefixes from this file')
    extractors.add_argument('--gazetteer', type=str,
                        help='Load gazetteer from this file')
    extractors.add_argument('--gsize', type=int, default=5,
                        help='Size of gazetteer features (default %(default)s)')

    # reader
    parser.add_argument('--form-field', type=int, default=0,
                        help='Token field containing form (default %(default)s)',
                        dest='formField')

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
        reader = NerReader(args.formField)

        # a generator (can be iterated several times)
        sentence_iter = reader.read(args.train)

        if args.vocab and os.path.exists(args.vocab):
            if args.vectors and os.path.exists(args.vectors):
                # use supplied embeddings
                embeddings = Embeddings(vectors=args.vectors, vocab_file=args.vocab,
                                        variant=args.variant)
            else:
                # create random embeddings
                embeddings = Embeddings(args.embeddings_size, vocab_file=args.vocab,
                                        variant=args.variant)
            # add the ngrams from the corpus
            # build vocabulary and tag set
            if args.vocab_size:
                vocab, tagset = reader.create_vocabulary(sentence_iter,
                                                         args.vocab_size,
                                                         args.minOccurr)
                embeddings.merge(vocab)
                logger.info("Overriding vocabulary in %s" % args.vocab)
                embeddings.save_vocabulary(args.vocab)
            else:
                tagset = reader.create_tagset(sentence_iter)

        elif args.variant == 'word2vec':
            if os.path.exists(args.vectors):
                embeddings = Embeddings(vectors=args.vectors,
                                        variant=args.variant)
                vocab, tagset = reader.create_vocabulary(sentence_iter,
                                                         args.vocab_size,
                                                         args.minOccurr)
                embeddings.merge(vocab)
            else:
                vocab, tagset = reader.create_vocabulary(sentence_iter,
                                                         args.vocab_size,
                                                         args.minOccurr)
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
            vocab, tagset = reader.create_vocabulary(sentence_iter,
                                                     args.vocab_size,
                                                     args.minOccurr)
            logger.info("Creating word embeddings")
            embeddings = Embeddings(args.embeddings_size, vocab=vocab,
                                    variant=args.variant)
            if args.vocab:
                logger.info("Saving vocabulary in %s" % args.vocab)
                embeddings.save_vocabulary(args.vocab)

        converter = Converter()
        # pass just the formField from tokens to the extractor
        converter.add(embeddings, reader.formField)
        
        if args.caps:
            logger.info("Creating capitalization features...")
            converter.add(CapsExtractor(args.caps), reader.formField)

        if args.pos:
            logger.info("Creating POS features...")
            postags = frozenset((token[args.pos] for sent in sentence_iter for token in sent))
            # tell the extractor which field to use 
            converter.add(AttributeExtractor(postags), args.pos) # no variant, preserve case

        if ((args.suffixes and not os.path.exists(args.suffixes)) or
            (args.prefixes and not os.path.exists(args.prefixes))):
            # collect the forms once
            words = (tok[reader.formField] for sent in sentence_iter for tok in sent)

        if args.suffix:
            if os.path.exists(args.suffixes):
                logger.info("Loading suffix list...")
                extractor = SuffixExtractor(args.suffix, args.suffixes)
                converter.add(extractor, reader.formField)
            else:
                logger.info("Creating suffix list...")
                extractor = SuffixExtractor(args.suffix, None, words)
                converter.add(extractor, reader.formField)
                if args.suffixes:
                    logger.info("Saving suffix list to: %s", args.suffixes)
                    extractor.write(args.suffixes)

        if args.prefix:
            if os.path.exists(args.prefixes):
                logger.info("Loading prefix list...")
                extractor = PrefixExtractor(args.prefix, args.prefixes)
                converter.add(extractor, reader.formField)
            else:
                logger.info("Creating prefix list...")
                extractor = PrefixExtractor(args.prefix, None, words)
                converter.add(extractor, reader.formField)
                if args.prefixes:
                    logger.info("Saving prefix list to: %s", args.prefixes)
                    extractor.write(args.prefixes)

        if args.gazetteer:
            if os.path.exists(args.gazetteer):
                logger.info("Loading gazetteers")
                for extractor in GazetteerExtractor.create(args.gazetteer, args.gsize):
                    # tell the extractor which field to use 
                    converter.add(extractor, reader.formField)
            else:
                logger.info("Creating gazetteer")
                tries = GazetteerExtractor.build(sentence_iter, reader.formField, reader.tagField)
                for tag, trie in tries.items():
                    # tell the extractor which field to use 
                    converter.add(GazetteerExtractor(trie, args.gsize), reader.formField)
                logger.info("Saving gazetteer list to: %s", args.gazetteer)
                with open(args.gazetteer, 'wb') as file:
                    for tag, trie in tries.iteritems():
                        for ngram in trie:
                            print(('%s\t%s' % (tag, ' '.join(ngram))).encode('UTF-8'), file=file)

        # if args.pos:
        #     converter.add(POS(arg.pos))

        # obtain the tags for each sentence
        tag_index = { t:i for i,t in enumerate(tagset) }
        sentences = []
        tags = []
        for sent in sentence_iter:
            sentences.append(converter.convert(sent))
            tags.append(np.array([tag_index[token[reader.tagField]] for token in sent],
                                 np.int32))
        logger.info("Vocabulary size: %d" % embeddings.dict.size())
        logger.info("Tagset size: %d" % len(tagset))
        trainer = create_trainer(args, converter, tag_index)
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
        reader = TaggerReader()
        for sent in reader:
            ConllWriter.write(tagger.tag(sent, reader.tagField))

# ----------------------------------------------------------------------

profile = False

if __name__ == '__main__':
    if profile:
        import cProfile
        cProfile.runctx("main()", globals(), locals(), "Profile.prof")
    else:
        main()
