#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
Script to test the networks on NLP tasks. 

* For POS , it will compute the accuracy.
* For SRL, it depends on the subtask:
    - Argument identification - Precision, recall and F1
    - Argument classification - Accuracy
    - Predicate detection - TP, TN, FP, FN and accuracy
    - Joint task - it will output the data in CoNLL-style to be 
evaluated externally.
"""

import logging
import argparse
from itertools import izip
import numpy as np
from collections import Counter, defaultdict

import nlpnet.config as config
import nlpnet.utils as utils
import nlpnet.taggers as taggers
from nlpnet.metadata import Metadata

def evaluate_pos(gold_file=None, oov=None):
    """
    Tests the network for tagging a given sequence.
    
    :param gold_file: file with gold data to evaluate against
    :param oov: either None or a list of tokens, that should contain the oov words.
    """
    md = Metadata.load_from_file('pos')
    nn = taggers.load_network(md)
    pos_reader = taggers.create_reader(md, gold_file=gold_file)
    itd = pos_reader.get_inverse_tag_dictionary()
    
    logger = logging.getLogger("Logger")
    logger.debug('Loaded network')
    logger.debug(nn.description())
    logger.info('Starting test...')
    hits = 0
    total = 0
    #pos_reader.codify_sentences()
    
    for sent in pos_reader.sentences:
        
        tokens, tags = zip(*sent)
        sent_codified = np.array([pos_reader.converter.convert(t) for t in tokens])
        answer = nn.tag_sentence(sent_codified)
        if oov is not None:
            iter_sent = iter(tokens)
        
        for net_tag, gold_tag in zip(answer, tags):
            
            if oov is not None:
                # only check oov words
                word = iter_sent.next()
                if word.lower() not in oov:
                    continue
            
            if itd[net_tag] == gold_tag:
                hits += 1
            
            total += 1                
        
    print '%d hits out of %d' % (hits, total)
    accuracy = float(hits) / total
    logger.info('Done.')
    return accuracy

def sentence_precision(network_tags, gold_tags, gold_tag_dict, network_tag_dict):
    """
    Evaluates the network precision on a single sentence.
    
    :param network_tags: the answers by the network
    :param gold_tags: the correct tags
    :param gold_tag_dict: inverse tag dictionary (numbers to tags) for the
        gold tags.
    :param network_tag_dict: inverse tag dictionary (numbers to tags) for the
        network answers.
    :returns: a tuple where the first member is the list of arguments
        tagged right and the second is the list of arguments found by
        the network.
    """
    inside_argument = False
    mistake = False
    cur_tag = ''
    correct_args = []
    predicted_args = []
    
    for net_tag, gold_tag in izip(network_tags, gold_tags):
        net_tag = network_tag_dict[net_tag]
        gold_tag = gold_tag_dict[gold_tag]
        
        # argument boundary identification uses IOBES. Convert it to IOB
        # to make evaluation easier
        if net_tag == 'E':
            net_tag = 'I'
        elif net_tag == 'S':
            net_tag = 'B'
        
        if inside_argument and net_tag[0] != 'I':
            # last argument ended
            predicted_args.append(cur_tag)
            if not mistake and gold_tag[0] != 'I':
                correct_args.append(cur_tag)
            inside_argument = False
        
        if net_tag[0] == 'B' and net_tag != 'B-V':
            inside_argument = True
            mistake = False
            cur_tag = ''
            
            # let's see if it's actually an argument and not the verb
            if gold_tag == 'B-V':
                inside_argument = False
            elif gold_tag[0] != 'B':
                mistake = True
            
        elif net_tag[0] == 'I' and inside_argument:
            if not mistake and gold_tag[0] != 'I':
                mistake = True
    
    # if there was still an argument
    if inside_argument:
        predicted_args.append(cur_tag)
        if not mistake:
            correct_args.append(cur_tag)
        
    return (correct_args, predicted_args)
        
def join_2_steps(boundaries, arguments):
    """
    Joins the tags for argument boundaries and classification accordingly.
    """
    answer = []
    
    for pred_boundaries, pred_arguments in izip(boundaries, arguments):
        cur_arg = ''
        pred_answer = []
        
        for boundary_tag in pred_boundaries:
            if boundary_tag == 'O':
                pred_answer.append('O')
            elif boundary_tag in 'BS':
                cur_arg = pred_arguments.pop(0)
                tag = '%s-%s' % (boundary_tag, cur_arg)
                pred_answer.append(tag)
            else:
                tag = '%s-%s' % (boundary_tag, cur_arg)
                pred_answer.append(tag)
        
        answer.append(pred_answer)
    
    return answer


def sentence_recall(network_tags, gold_tags, gold_tag_dict, network_tag_dict):
    """
    Evaluates the network recall on a single sentence.
    
    :param network_tags: the answers by the network
    :param gold_tags: the correct tags
    :param gold_tag_dict: inverse tag dictionary (numbers to tags) for the
    gold tags.
    :param network_tag_dict: inverse tag dictionary (numbers to tags) for the
    network answers.
    :returns: a tuple where the first member is the list of arguments
        got right and the second is the list of arguments that were in
        the sentence.
    """
    inside_argument = False
    mistake = False
    cur_tag = ''
    correct_args = []
    existing_args = []
    
    for net_tag, gold_tag in izip(network_tags, gold_tags):
        net_tag = network_tag_dict[net_tag]
        gold_tag = gold_tag_dict[gold_tag]
        
        # argument boundary identification uses IOBES. Convert it to IOB
        # to make evaluation easier
        if net_tag == 'E':
            net_tag = 'I'
        elif net_tag == 'S':
            net_tag = 'B'
    
        if inside_argument and gold_tag[0] != 'I':
            # last argument ended
            existing_args.append(cur_tag)
            inside_argument = False
            if not mistake and net_tag[0] != 'I':
                correct_args.append(cur_tag)
            mistake = False
        
        if gold_tag[0] == 'B' and gold_tag != 'B-V':
            # a new argument starts here
            inside_argument = True
            cur_tag = gold_tag[2:]
            if net_tag != 'B':
                mistake = True
        
        elif gold_tag[0] == 'I' and gold_tag != 'I-V':
            # inside an argument
            if not mistake and net_tag[0] != 'I':
                mistake = True
        
        # the network answer doesn't matter to recall on other cases ("O")
        
    # verifies if last argument ended with the sentence
    if inside_argument:
        # last argument ended
        existing_args.append(cur_tag)
        if not mistake:
            correct_args.append(cur_tag)
    
    return (correct_args, existing_args)

def evaluate_srl_classify(no_repeat=False, gold_file=None):
    """Evaluates the performance of the network on the SRL classifying task."""
    # load data
    md = Metadata.load_from_file('srl_classify')
    nn = taggers.load_network(md)
    r = taggers.create_reader(md, gold_file)
    r.create_converter(md)
    
    r.codify_sentences()
    hits = 0
    total_args = 0
    
    for sentence, tags, predicates, args in izip(r.sentences, r.tags, r.predicates, r.arg_limits):
        
        # the answer includes all predicates
        answer = nn.tag_sentence(sentence, predicates, args, allow_repeats=not no_repeat)
        
        for pred_answer, pred_gold in izip(answer, tags):
        
            for net_tag, gold_tag in izip(pred_answer, pred_gold):
                if net_tag == gold_tag:
                    hits += 1
            
            total_args += len(pred_gold)
    
    print 'Accuracy: %f' % (float(hits) / total_args)

def convert_iob_to_iobes(iob_tags):
    """Converts a sequence of IOB tags into IOBES tags."""
    iobes_tags = []
    
    # check each tag and its following one. A None object is appended 
    # to the end of the list
    for tag, next_tag in zip(iob_tags, iob_tags[1:] + [None]):
        if tag == 'O':
            iobes_tags.append('O')
        elif tag.startswith('B'):
            if next_tag is not None and next_tag.startswith('I'):
                iobes_tags.append(tag)
            else:
                iobes_tags.append('S-%s' % tag[2:])
        elif tag.startswith('I'):
            if next_tag == tag:
                iobes_tags.append(tag)
            else:
                iobes_tags.append('E-%s' % tag[2:])
        else:
            raise ValueError("Unknown tag: %s" % tag)
    
    return iobes_tags


def prop_conll(verbs, props, sent_length):
    """
    Returns the string representation for a single sentence
    using the CoNLL format for evaluation.
    
    :param verbs: list of tuples (position, token)
    :param props: list of lists with IOBES tags.
    """
    # defaultdict to know what to print in the verbs column
    verb_dict = defaultdict(lambda: '-', verbs)
    lines = []
    
    for i in range(sent_length):
        verb = verb_dict[i]
        args = [utils.convert_iobes_to_bracket(x[i]) for x in props]
        lines.append('\t'.join([verb] + args))
    
    # add a line break at the end
    result = '%s\n' % '\n'.join(lines) 
    return result.encode('utf-8')
        
def evaluate_srl_2_steps(no_repeat=False, find_preds_automatically=False, gold_file=None):
    """Prints the output of a 2-step SRL system in CoNLL style for evaluating."""
    # load boundary identification network and reader 
    md_boundary = Metadata.load_from_file('srl_boundary')
    nn_boundary = taggers.load_network(md_boundary)
    reader_boundary = taggers.create_reader(md_boundary, gold_file)
    itd_boundary = reader_boundary.get_inverse_tag_dictionary()
    
    # same for arg classification
    md_classify = Metadata.load_from_file('srl_classify')
    nn_classify = taggers.load_network(md_classify)
    reader_classify = taggers.create_reader(md_classify, gold_file)
    itd_classify = reader_classify.get_inverse_tag_dictionary()
    
    if find_preds_automatically:
        tagger = taggers.SRLTagger()
    else:
        iter_predicates = iter(reader_boundary.predicates)
    
    actual_sentences = [actual_sentence for actual_sentence, _ in reader_boundary.sentences]
    
    for sent in actual_sentences:
        
        if find_preds_automatically:
            pred_pos = tagger.find_predicates(sent)
        else:
            pred_pos = iter_predicates.next()
        
        verbs = [(position, sent[position].word) for position in pred_pos]
        sent_bound_codified = np.array([reader_boundary.converter.convert(t) for t in sent])
        sent_class_codified = np.array([reader_classify.converter.convert(t) for t in sent])
        
        answers = nn_boundary.tag_sentence(sent_bound_codified, pred_pos)
        boundaries = [[itd_boundary[x] for x in pred_answer] for pred_answer in answers]
        
        arg_limits = [utils.boundaries_to_arg_limits(pred_boundaries) 
                      for pred_boundaries in boundaries]
        
        answers = nn_classify.tag_sentence(sent_class_codified, 
                                           pred_pos, arg_limits,
                                           allow_repeats=not no_repeat)
        
        arguments = [[itd_classify[x] for x in pred_answer] for pred_answer in answers]        
        tags = join_2_steps(boundaries, arguments)        
        
        print prop_conll(verbs, tags, len(sent))

def evaluate_srl_1step(find_preds_automatically=False, gold_file=None):
    """
    Evaluates the network on the SRL task performed with one step for
    id + class.
    """
    md = Metadata.load_from_file('srl')
    nn = taggers.load_network(md)
    r = taggers.create_reader(md, gold_file=gold_file)
    
    itd = r.get_inverse_tag_dictionary()
    
    if find_preds_automatically:
        tagger = taggers.SRLTagger()
    else:
        iter_predicates = iter(r.predicates)
    
    for sent in iter(r.sentences):
        
        # the other elements in the list are the tags for each proposition
        actual_sent = sent[0]
        
        if find_preds_automatically:
            pred_positions = tagger.find_predicates(sent)
        else:
            pred_positions = iter_predicates.next()
            
        verbs = [(position, actual_sent[position].word) for position in pred_positions]
        sent_codified = np.array([r.converter.convert(token) for token in actual_sent])
        
        answers = nn.tag_sentence(sent_codified, pred_positions)
        tags = [convert_iob_to_iobes([itd[x] for x in pred_answer])
                for pred_answer in answers]
            
        print prop_conll(verbs, tags, len(actual_sent))
        
def evaluate_srl_predicates(gold_file):
    """
    Evaluates the performance of the network on the SRL task for the
    predicate detection subtask.
    """
    md = Metadata.load_from_file('srl_predicates')
    nn = taggers.load_network(md)
    reader = taggers.create_reader(md, gold_file=gold_file)
    reader.codify_sentences()
    
    total_tokens = 0
    # true/false positives and negatives
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # for each sentence, tags are 0 at non-predicates and 1 at predicates
    for sent, tags in izip(reader.sentences, reader.tags):
        answer = nn.tag_sentence(sent)
        
        for net_tag, gold_tag in izip(answer, tags):
            if gold_tag == 1:
                if net_tag == gold_tag: tp += 1
                else: fn += 1
            else:
                if net_tag == gold_tag: tn += 1
                else: fp += 1
        
        total_tokens += len(sent)
    
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    
    print 'True positives: %d, false positives: %d, \
true negatives: %d, false negatives: %d' % (tp, fp, tn, fn)
    print 'Accuracy: %f' % (float(tp + tn) / total_tokens) 
    print 'Precision: %f' % precision
    print 'Recall: %f' % recall
    print 'F-1: %f' % (2 * precision * recall / (precision + recall))
    
    
def evaluate_srl_identify(gold_file):
    """
    Evaluates the performance of the network on the SRL task for the 
    argument boundaries identification subtask
    """
    md = Metadata.load_from_file('srl_boundary')
    nn = taggers.load_network(md)
    srl_reader = taggers.create_reader(md, gold_file=gold_file)
    
    net_itd = srl_reader.get_inverse_tag_dictionary()
    srl_reader.load_tag_dict(config.FILES['srl_iob_tag_dict'])
    
    srl_reader.convert_tags('iob', update_tag_dict=False)
    gold_itd = srl_reader.get_inverse_tag_dictionary()
 
    # used for calculating precision
    counter_predicted_args = Counter()
    # used for calculating recall
    counter_existing_args = Counter()
    # used for calculating both
    counter_correct_args = Counter()

    srl_reader.codify_sentences()
    
    for sent, preds, sent_tags in izip(srl_reader.sentences, srl_reader.predicates, srl_reader.tags):
        
        # one answer for each predicate
        answers = nn.tag_sentence(sent, preds)
        
        for answer, tags in zip(answers, sent_tags):
            correct_args, existing_args = sentence_recall(answer, tags, gold_itd, net_itd)
            counter_correct_args.update(correct_args)
            counter_existing_args.update(existing_args)
            
            _, predicted_args = sentence_precision(answer, tags, gold_itd, net_itd)
            counter_predicted_args.update(predicted_args)
            
    correct_args = sum(counter_correct_args.values())
    total_args = sum(counter_existing_args.values())
    total_found_args = sum(counter_predicted_args.values())
    rec = correct_args / float(total_args)
    prec = correct_args / float(total_found_args)
    try:
        f1 = 2 * rec * prec / (rec + prec)
    except ZeroDivisionError:
        f1 = 0

    print 'Recall: %f, Precision: %f, F-1: %f' % (rec, prec, f1)
    print
    print 'Argument\tRecall'
    
    for arg in counter_existing_args:
        rec = counter_correct_args[arg] / float(counter_existing_args[arg])
        
        # a couple of notes about precision per argument:
        # - we can't compute it if we are only interested in boundaries. hence, we can't compute f-1
        # - if the network never tagged a given argument, its precision is 100% (it never made a mistake)
                
        print '%s\t\t%f' % (arg, rec)

def read_oov_words(oov_file):
    words = set()
    with open(oov_file, 'rb') as f:
        for line in f:
            uline = unicode(line, 'utf-8').strip().lower()
            words.add(uline)
    
    return words

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task for which the network should be used.', 
                        type=str, required=True, choices=['srl', 'pos'])
    parser.add_argument('-v', help='Verbose mode', action='store_true', dest='verbose')
    parser.add_argument('--id', help='Evaluate only argument identification (SRL only)',
                        action='store_true', dest='identify')
    parser.add_argument('--class', help='Evaluate only argument classification (SRL only)',
                        action='store_true', dest='classify')
    parser.add_argument('--preds', help='Evaluate only predicate identification (SRL only)',
                        action='store_true', dest='predicates')
    parser.add_argument('--2steps', help='Execute SRL with two separate steps', action='store_true', dest='two_steps')
    parser.add_argument('--no-repeat', dest='no_repeat', action='store_true',
                        help='Forces the classification step to avoid repeated argument labels (2 step SRL only).')
    parser.add_argument('--auto-pred', dest='auto_pred', action='store_true',
                        help='Determines SRL predicates automatically using a POS tagger.')
    parser.add_argument('--gold', help='File with gold standard data', type=str, required=True)
    parser.add_argument('--data', help='Directory with trained models', type=str, required=True)
    parser.add_argument('--oov', help='Analyze performance on OOV data', type=str)
    args = parser.parse_args()
    
    if args.identify:
        args.task = 'srl_boundary'
    elif args.classify:
        args.task = 'srl_classify'
    elif args.predicates:
        args.task = 'srl_predicates'
    
    logging_level = logging.DEBUG if args.verbose else logging.WARNING
    utils.set_logger(logging_level)
    logger = logging.getLogger("Logger")
    config.set_data_dir(args.data)
    
    if args.task == 'pos':
        
        if args.oov:
            oov = read_oov_words(args.oov)
        else:
            oov = None
                    
        accuracy = evaluate_pos(gold_file=args.gold, oov=oov)
        print "Accuracy: %f" % accuracy
    
    elif args.task.startswith('srl'):
        
        if args.oov:
            logger.error('OOV not implemented for SRL.')
        
        if args.two_steps:
            evaluate_srl_2_steps(args.no_repeat, args.auto_pred, args.gold)
        elif args.classify:
            evaluate_srl_classify(args.no_repeat, args.gold)
        elif args.identify:
            evaluate_srl_identify(args.gold)
        elif args.predicates:
            evaluate_srl_predicates(args.gold)
        else:
            evaluate_srl_1step(args.auto_pred, args.gold)
        
