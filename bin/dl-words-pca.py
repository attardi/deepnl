#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learn word embeddings from plain text using Hellinger PCA.

See
Lebret, Rémi, and Ronan Collobert. "Word Embeddings through Hellinger PCA." EACL 2014 (2014): 482.

Author: Giuseppe Attardi
"""

import logging
import numpy as np
import argparse
from ConfigParser import ConfigParser

# profiling
# import yappi
# cProfile
# import pstats, cProfile
# import pyximport
# pyximport.install()

# allow executing from anywhere without installing the package
import sys
import os
import distutils.util
builddir = os.path.dirname(os.path.realpath(__file__)) + '/../build/lib.'
libdir = builddir + distutils.util.get_platform() + '-' + '.'.join(map(str, sys.version_info[:2]))
sys.path.append(libdir)

# local
from deepnl.embeddings import Plain
import deepnl.hpca as hpca

# ----------------------------------------------------------------------

def main():

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
                        help='Size of the word window (default %(default)s)')
    parser.add_argument('-s', '--embeddings-size', type=int, default=50,
                        help='Number of features per word (default %(default)s)',
                        dest='embeddings_size')
    parser.add_argument('--ngrams', type=int, default=1,
                        help='Size of ngrams (default %(default)s)')
    parser.add_argument('--train', type=str, required=True,
                        help='File with text corpus for training.')
    parser.add_argument('-o', '--output', type=str,
                        help='File where to save the model, for further training')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Vocabulary file')
    parser.add_argument('--context-words', type=int, default=10000,
                        help='Number of context words (the first N from vocabulary)')
    parser.add_argument('--context-size', type=int, default=1,
                        help='Number of context words')
    parser.add_argument('--vectors', type=str, required=True,
                        help='Embeddings file, either read and updated or created')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads (default %(default)s)')
    parser.add_argument('--variant', type=str, default=None,
                        help='Either "senna" (default), "polyglot" or "word2vec".')
    parser.add_argument('--covariance', action='store_true',
                        help='Use PCA algorithm on covariance matrix.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose mode')

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

    logger.info("Building co-occurrence matrix")
    rootmat = hpca.cooccurrences(args.train, args.vocab, args.context_words,
                                 args.context_size)
    logger.info("Perform PCA")
    vectors = hpca.fit(rootmat, args.embeddings_size, args.covariance)

    logger.info("Saving vectors ...")
    Plain.write_vectors(args.vectors, vectors)
    logger.info("... to %s" % args.vectors)

    if args.output:
        logger.info("Saving trained model ...")
        trainer.save(args.output)
        logger.info("... to %s" % args.output)

# ----------------------------------------------------------------------

profile = None #'yappi'

if __name__ == '__main__':
    if profile == 'yappi':
        yappi.start()
        main()
        yappi.get_func_stats().print_all()
    elif profile == 'cprofile':
        cProfile.runctx("main()", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        main()
