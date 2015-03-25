#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train a neural network for NLP tagging tasks.

Author: Erick Rocha Fonseca and Giuseppe Attardi
"""

import logging
import numpy as np
from ConfigParser import ConfigParser

# allow executing from anywhere without installing the package
import sys
import os
import distutils.util
builddir = os.path.dirname(os.path.realpath(__file__)) + '/../build/lib.'
libdir = builddir + distutils.util.get_platform() + '-' + '.'.join(map(str, sys.version_info[:2]))
sys.path.append(libdir)

# local
from nlpnet import *
import nlpnet.arguments as arguments
import nlpnet.extractors
import nlpnet.reader as reader
import nlpnet.utils as utils
from nlpnet.network import Network
from nlpnet.networklm import LanguageModel
from nlpnet.networkconv import ConvolutionalNetwork
from nlpnet.networkSent import SentimentModel

############################
### FUNCTION DEFINITIONS ###
############################

def create_reader(args, config):
    """
    Creates and returns a TextReader object according to the task at hand.
    """
    logger.info("Reading text...")
    if args.task == 'pos':
        reader = pos.pos_reader.POSReader(config, filename=args.gold, variant=args.variant)
        if args.suffix:
            reader.create_suffix_list(args.suffix_size, 5)
        if args.prefix:
            reader.create_prefix_list(args.prefix_size, 5)

    elif args.task == 'ner':
        reader = ner.ner_reader.NerReader(config, filename=args.gold,
                                          variant=args.variant)

    elif args.task == 'lm':
        reader = reader.TextReader(config, filename=args.gold,
                                   variant=args.variant)

    elif args.task == 'sslm':
        reader = reader.TweetReader(config, filename=args.gold,
                                    ngrams=args.ngrams, variant=args.variant)

    elif args.task.startswith('srl'):
        reader = srl.srl_reader.SRLReader(config, filename=args.gold,
                                          only_boundaries=args.identify, 
                                          only_classify=args.classify,
                                          only_predicates=args.predicates,
                                          variant=args.variant)
    
        if args.identify:
            # only identify arguments
            reader.convert_tags('iobes', only_boundaries=True)
            
        elif not args.classify and not args.predicates:
            # this is SRL as one step, we use IOB
            reader.convert_tags('iob', update_tag_dict=False)
        
    else:
        raise ValueError("Unknown task: %s" % args.task)
    
    reader.get_dictionaries(args.dict_size)
    return reader
    

def create_network(args, reader, feature_tables, config=None):
    """
    Creates and returns the neural network according to the specified args.
    """

    logger = logging.getLogger("Logger")

    if args.task.startswith('srl') and args.task != 'srl_predicates':
        num_tags = len(reader.tag_dict)
        distance_tables = utils.set_distance_features(args.max_dist, args.target_features,
                                                      args.pred_features)
        nn = ConvolutionalNetwork.create_new(feature_tables, distance_tables[0], 
                                             distance_tables[1], args.window, 
                                             args.convolution, args.hidden, num_tags)
        padding_left = reader.converter.get_padding_left(False)
        padding_right = reader.converter.get_padding_right(False)
        if args.identify:
            logger.info("Loading initial transition scores table for argument identification")
            nn.transitions = srl.train_srl.init_transitions_simplified(reader.tag_dict)
            nn.learning_rate_trans = args.learning_rate_transitions
            
        elif not args.classify:
            logger.info("Loading initial IOB transition scores table")
            nn.transitions = srl.train_srl.init_transitions(reader.tag_dict, 'iob')
            nn.learning_rate_trans = args.learning_rate_transitions
    
    elif args.task == 'lm':
        nn = LanguageModel.create_new(feature_tables, args.window/2,
                                      args.window/2, args.hidden)
        padding_left = reader.converter.get_padding_left(tokens_as_string=True)
        padding_right = reader.converter.get_padding_right(tokens_as_string=True)

    elif args.task == 'sslm':
        nn = SentimentModel.create_new(feature_tables, args.window/2,
                                       args.windows/2, args.hidden, args.alpha)
        padding_left = reader.converter.get_padding_left(tokens_as_string=True)
        padding_right = reader.converter.get_padding_right(tokens_as_string=True)
        nn.threads = args.threads
        
    else:
        # pos, srl_predicates or ner
        num_tags = len(reader.tag_dict)
        nn = Network.create_new(feature_tables, args.window/2, args.window/2,
                                args.hidden, num_tags)

        if args.learning_rate_transitions > 0:
            nn.transitions = np.zeros((num_tags + 1, num_tags), np.float)
            nn.learning_rate_trans = args.learning_rate_transitions

        padding_left = reader.converter.get_padding_left(args.task == 'pos' or args.task == 'ner')
        padding_right = reader.converter.get_padding_right(args.task == 'pos' or args.task == 'ner')
    
    nn.padding_left = np.array(padding_left)
    nn.padding_right = np.array(padding_right)
    nn.learning_rate = args.learning_rate
    nn.learning_rate_features = args.learning_rate_features
    
    if args.task == 'lm':
        layer_sizes = (nn.input_size, nn.hidden_size, 1)
    elif args.task == 'sslm':
        layer_sizes = (nn.input_size, nn.hidden_size, 2)
    elif 'convolution' in args and args.convolution > 0 and args.hidden > 0:
        layer_sizes = (nn.input_size, nn.hidden_size, nn.hidden2_size, nn.output_size)
    else:
        layer_sizes = (nn.input_size, nn.hidden_size, nn.output_size)
    
    logger.info("Created new network with the following layer sizes: %s" %
                ', '.join(map(str, layer_sizes)))
    
    return nn
        
def save_features(nn, config):
    """
    Receives a sequence of feature tables and saves each one in the
    appropriate file.
    
    :param nn: the neural network
    :param config: a Metadata object describing the network

    """
    def save_affix_features(affix, iter_tables):
        """
        Helper function for both suffixes and affixes.
        affix should be either 'suffix' or 'affix'
        """
        # there can be an arbitrary number of tables, one for each length
        affix_features = []
        codes = getattr(attributes.Affix, '%s_codes' % affix)
        num_sizes = len(codes)
        for _ in range(num_sizes):
            affix_features.append(iter_tables.next())
        
        filename_key = getattr(md, '%s_features' % affix)
        filename = config.FILES[filename_key]
        utils.save_features_to_file(affix_features, filename)

    iter_tables = iter(nn.feature_tables)
    # type features
    utils.save_features_to_file(iter_tables.next(), config.FILES[md.type_features])
    
    # other features - the order is important!
    if md.use_caps: utils.save_features_to_file(iter_tables.next(), config.FILES[md.caps_features])
    if md.use_prefix:
        save_affix_features('prefix', iter_tables)
    if md.use_suffix:
        save_affix_features('suffix', iter_tables)
    if md.use_pos: utils.save_features_to_file(iter_tables.next(), config.FILES[md.pos_features])
    if config.use_chunk: utils.save_features_to_file(iter_tables.next(), config.FILES[config.chunk_features])

    # NER gazetteer features
    if config.use_gazetteer:
        for file in config.FILES[config.gaz_features]:
            utils.save_features_to_file(iter_tables.next(), file)
    
def load_network_train(args, config):
    """Loads and returns a neural network with all the necessary data."""
    nn = taggers.load_network(config)
    
    logger.info("Loaded network with following parameters:")
    logger.info(nn.description())
    
    nn.learning_rate = args.learning_rate
    nn.learning_rate_features = args.learning_rate_features
    if config.task != 'lm' and config.task != 'sslm':
        nn.learning_rate_trans = args.learning_rate_transitions
    
    return nn

def create_metadata(args):
    """Creates a Metadata object from the given arguments."""
    # using getattr because the SRL args object doesn't have a "suffix" attribute
    use_caps = getattr(args, 'caps', False)
    use_suffix = getattr(args, 'suffix', False)
    use_prefix = getattr(args, 'prefix', False)
    use_pos = getattr(args, 'pos', False)
    use_chunk = getattr(args, 'chunk', False)
    use_lemma = getattr(args, 'lemma', False)
    use_gazetteer = getattr(args, 'gazetteer', False)
    
    return metadata.Metadata(args.task, None, use_caps, use_suffix, use_prefix, 
                             use_pos, use_chunk, use_lemma, use_gazetteer)

def train(nn, reader, args):
    """Trains a neural network for the selected task."""
    report_intervals = max(args.iterations / 200, 1)
    np.seterr(over='raise')
    
    if args.task.startswith('srl') and args.task != 'srl_predicates':
        arg_limits = None if args.task != 'srl_classify' else reader.arg_limits
        
        nn.train(reader.sentences, reader.predicates, reader.tags, 
                 args.iterations, report_intervals, args.accuracy, arg_limits)
    elif args.task == 'lm':
        report_intervals = 10000
        nn.train(reader.sentences, args.iterations, report_intervals, args.threads)
    elif args.task == 'sslm':
        report_intervals = 10000
        nn.train(reader.sentences, args.iterations, report_intervals, reader.polarities, reader.word_dict)
    else:                       # pos, ner
        nn.train(reader.sentences, reader.tags, 
                 args.iterations, report_intervals, args.accuracy)

def saver(nn_file):
    """Function to save model periodically"""
    def save(nn):
        save_features(nn, md)
        nn.save(nn_file)
    return save

classForTask = {
    'lm': LanguageModel,
    'sslm': SentimentModel,
    'pos': PosTagger,
    'ner': NerTagger,
    'srl': {'pred': Network,
            'id': ConvolutionalNetwork,
            'class': ConvolutionalNetwork,
            '1step': ConvolutionalNetwork
            }
}

if __name__ == '__main__':
    args = arguments.get_args()

    # set the seed for replicability
    #np.random.seed(42)

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    utils.set_logger(logging_level)
    logger = logging.getLogger("Logger")

    config = ConfigParser()
    if args.config_file:
        config.read(args.config_file)

    # merge args with config

    # determine which class to create
    cls = classForTask[config.task]

    reader = cls.create_reader(config)
    
    converter = cls.create_converter(config)

    reader.codify_sentences(converter)
    
    if args.load_network:
        logger.info("Loading provided network...")
        nn = load_network_train(args, md)
    else:
        logger.info('Creating new network...')
        feature_tables = utils.create_feature_tables(args, md, reader)
        nn = create_network(args, reader, feature_tables, md)
    
    nn.filename = config.FILES[md.network]
    nn.saver = saver(nn.filename, md)

    logger.info("Starting training with %d sentences" % len(reader.sentences))
    logger.info("Network weights learning rate: %f" % nn.learning_rate)
    logger.info("Feature vectors learning rate: %f" % nn.learning_rate_features)
    if nn.learning_rate_trans:
        logger.info("Tag transition matrix learning rate: %f" % nn.learning_rate_trans)

    train(nn, reader, args)
    
    logger.info("Saving trained models...")
    save_features(nn, md)
    
    nn.save(nn.filename)
    logger.info("Saved network to %s" % nn_file)
    
