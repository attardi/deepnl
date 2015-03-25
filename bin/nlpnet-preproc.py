#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collection of preprocessing tools for nlpnet.
"""

import argparse

from nlpnet.pos.pos_reader import POSReader
from nlpnet.srl.srl_reader import SRLReader

def parse_args():
    '''
    Deal with argument stuff.
    '''
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(title='Tools')
    
    gen_dict_help = 'Generate and save (as pickle) '\
                    'a word dictionary for the given corpus'
    gen_dict = subparsers.add_parser('generate-dict',
                                     description=gen_dict_help,
                                     help=gen_dict_help)
    gen_dict.add_argument('corpus', help='Corpus file')
    gen_dict.add_argument('task', choices=['pos', 'srl'],
                          help='Task the corpus is intended for.'
                          'POS corpora should have lines in the format '
                          'token_tag token_tag ...'
                          'SRL corpora should be in CoNLL format.')
    gen_dict.add_argument('minimum_occurrences',
                          help='Minimum number of occurrences of a word in the corpus '
                          'to be included in the dictionary', type=int)
    gen_dict.add_argument('output', help='File to save the dictionary')
    gen_dict.set_defaults(func=generate_dict)
    
    
    gen_tag_dict_help = 'Generate and save (as pickle) a tag '\
                        'dictionary from the given corpus'
    gen_tag_dict = subparsers.add_parser('generate-tag-dict',
                                         help=gen_tag_dict_help,
                                         description=gen_tag_dict_help)
    gen_tag_dict.add_argument('corpus', help='Corpus file')
    gen_tag_dict.add_argument('output', help='File to save the dictionary')
    gen_tag_dict.set_defaults(func=generate_tag_dict)
    
    args = parser.parse_args()
    return args


def get_reader_for_task(task):
    '''
    Return the Reader class corresponding to the given class.
    '''
    if args.task == 'pos':
        reader_class = POSReader
    elif args.task == 'srl':
        reader_class = SRLReader
    else:
        raise ValueError('Unknown task: %s' % args.task)
    
    return reader_class


def generate_tag_dict(args):
    '''
    Generate and save a tag dictionary from the corpus.
    '''
    reader_class = get_reader_for_task(args.task)
    r = reader_class(filename=args.corpus, load_dictionaries=False)
    r.generate_tag_dict()


def generate_dict(args):
    '''
    Generate and save a WordDictionary from the corpus.
    '''
    reader_class = get_reader_for_task(args.task)
    r = reader_class(filename=args.corpus, load_dictionaries=False)
    r.generate_dictionary(minimum_occurrences=args.minimum_occurrences)
    r.save_word_dict(args.output)


if __name__ == '__main__':

    args = parse_args()
    
    # call the default function to handle the tool
    args.func(args)
    




