#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run either a POS, NER or SRL tagger on the input data and
print the results to stdout.

"""
# standard
import argparse
import logging
from itertools import izip

# allow executing from anywhere without installing package
import sys
import os
import distutils.util
builddir = os.path.dirname(os.path.realpath(__file__)) + '/../build/lib.'
libdir = builddir + distutils.util.get_platform() + '-' + '.'.join(map(str, sys.version_info[:2]))
sys.path.append(libdir)

# local
import nlpnet
import nlpnet.utils as utils

from nlpnet.corpus import *

def process_input(task):
    """
    This function reads input from stdin and processes sentences.
    
    :param task: either 'pos', 'srl' or 'ner'
    """
    reader = ConllReader()
    writer = ConllWriter
    task_lower = task.lower()
    if task_lower == 'pos':
        tagger = nlpnet.taggers.POSTagger()
    elif task_lower == 'srl':
        tagger = nlpnet.taggers.SRLTagger()
        writer = SrlWriter
    elif task_lower == 'ner':
        tagger = nlpnet.taggers.NERTagger()
    else:
        raise ValueError('Unknown task: %s' % task)
    
    for sent in reader:
        writer.write(tagger.tag_tokens(sent, True))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task for which the network should be used.', 
                        type=str, choices=['srl', 'pos', 'ner'])
    parser.add_argument('data', help='Directory containing trained models.', type=str)
    parser.add_argument('-v', help='Verbose mode', action='store_true', dest='verbose')
    parser.add_argument('--no-repeat', dest='no_repeat', action='store_true',
                        help='Forces the classification step to avoid repeated argument labels (SRL only).')
    args = parser.parse_args()
    
    logging_level = logging.DEBUG if args.verbose else logging.WARNING
    utils.set_logger(logging_level)
    nlpnet.set_data_dir(args.data)
    
    process_input(args.task)

