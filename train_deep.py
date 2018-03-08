#!/usr/bin/env python3
"""
Train a DeepCNNRelScorer on a precomputed question-relations file

Niklas Schnelle <schnelle@cs.uni-freiburg.de>
"""
import logging
import argparse
import random
from collections import defaultdict

from deep_relscorer import DeepCNNRelScorer
import config_helper
from sklearn import utils


def extract_question_tokens(qstr):
    """
    Turns a question string including [<mid>|orig_text] mentions
    into a list of question tokens. It prepends a <start> token and
    replaces mentions with [entity]. Punctuation is ignored
    """
    tokens = ['<start>']
    for token in qstr.split(' '):
        if token[0] == '[' and token[-1] == ']':
            mid, _ = token[1:-1].split('|')
            placeholder = '[entity]'
            # always at least <start>
            if tokens[-1] != placeholder:
                tokens.append(placeholder)
        elif not token.isalnum():
            continue
        else:
            tokens.append(token.lower())
    return tuple(tokens)


def relation_examples_from_file(path):
    """
    Reads question - relation examples from a text file
    """
    pos_examples = []
    neg_examples = []
    with open(path, 'r', encoding='utf-8') as relations_file:
        for line in relations_file:
            question_str, _, pos_rels_str, neg_rels_str = line.split('\t')
            question_tokens = extract_question_tokens(question_str)
            pos_rels_str = pos_rels_str.strip()
            neg_rels_str = neg_rels_str.strip()

            pos_rels = [tuple(sorted(relation.split(',')))
                        for relation in pos_rels_str.split(' ')]
            neg_rels = [tuple(sorted(relation.split(',')))
                        for relation in neg_rels_str.split(' ')]

            for relation in pos_rels:
                pos_examples.append((question_tokens, relation))

            for relation in neg_rels:
                neg_examples.append((question_tokens, relation))
    return pos_examples, neg_examples


def read_category_map(path):
    """
    Reads a category map file of the form <mid>\t<category\n
    and returns it as a map from mid to category.
    """
    print('Read category map file')
    category_map = defaultdict(lambda: 'Unknown')
    with open(path, 'r', encoding='utf-8') as category_map_file:
        for line in category_map_file:
            splits = line.strip().split('\t')
            mid = splits[0]
            if len(splits) > 1:
                category_map[mid] = splits[1]
    print('Done reading category map file')
    return category_map

def main():
    """
    The main function handling arguments etc
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use.')
    parser.add_argument('-m', '--model-name', default='WQSP_ExtDeep_Ranker')
    parser.add_argument('-p', '--model-path', default='models')
    parser.add_argument('-e', '--extend-model', default=None)
    parser.add_argument('--dev-ratio', type=float, default=0.1)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--num-hidden-nodes', type=int, default=200)
    parser.add_argument('--num-filters', type=int, default=64)
    parser.add_argument('--no-attention', action='store_true', default=False)
    parser.add_argument('questions_file')

    args = parser.parse_args()
    random.seed(1312)

    config_helper.read_configuration(args.config)

    config_options = config_helper.config

    rel_model = DeepCNNRelScorer.init_from_config(
            use_type_names=False,
            use_attention=not args.no_attention,
            num_filters=args.num_filters,
            num_hidden_nodes=args.num_hidden_nodes)
    print('Reading questions')
    pos_examples, neg_examples = relation_examples_from_file(
        args.questions_file)
    num_pos = len(pos_examples)
    num_neg = len(neg_examples)

    # Create dev batches
    num_pos_dev = int(num_pos * args.dev_ratio)
    pos_dev = pos_examples[num_pos-num_pos_dev:]
    pos_examples = pos_examples[:num_pos-num_pos_dev]

    num_neg_dev = int(num_neg * args.dev_ratio)
    neg_dev = neg_examples[num_neg-num_neg_dev:]
    neg_examples = neg_examples[:num_neg-num_neg_dev]

    print('Dev examples: {} positive, {} negative'.format(
        num_pos_dev, num_neg_dev))

    dev_examples = pos_dev + neg_dev
    # hashing the question tokens (which are a tuple)
    # allows us to get a question id that matches queries with the same
    # tokens without using a lot of memory (mapping/set)
    dev_qids = [hash(toks) for toks, _ in dev_examples]
    dev_f1s = [1.0 for _ in range(num_pos_dev)] + \
        [0.0 for _ in range(num_neg_dev)]
    dev_examples, dev_qids, dev_f1s = \
        utils.shuffle(dev_examples, dev_qids, dev_f1s)


    rel_model.learn_relation_model(pos_examples, neg_examples,
                                   extend_model=args.extend_model,
                                   dev_examples=dev_examples,
                                   dev_qids=dev_qids,
                                   dev_f1s=dev_f1s,
                                   num_epochs=args.num_epochs)
    rel_model.store_model(args.model_path, args.model_name)

if __name__ == '__main__':
    main()
