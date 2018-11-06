from __future__ import division
import argparse
import cPickle
import itertools
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--doc_list",  help='comma-delimited list of processed documents', type=str)


def get_root_agreement(doc_list):
    doc_pairs = itertools.combinations(range(0,len(doc_list)), 2)
    all_agreements = []
    for doc_pair in doc_pairs:
        pair_agreements = []
        docs1 = doc_list[doc_pair[0]]
        docs2 = doc_list[doc_pair[0]]
        for doc1, doc2 in zip(docs1, docs2):
            if set(doc1.tree.deps[0]).intersection(set(doc2.tree.deps[0])):
                pair_agreements.append(1)
            else:
                pair_agreements.append(0)
        all_agreements.append(pair_agreements)
    all_agreements = np.array(all_agreements)
    avg_agreement = np.mean(np.sum(all_agreements, axis=1)) / all_agreements.shape[1]
    return avg_agreement


def main(_):
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data_structure import ProcessedDoc

    args = parser.parse_args()
    doc_list = [cPickle.load(open(item)) for item in args.doc_list.split(',')]
    get_root_agreement(doc_list)
