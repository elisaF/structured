from __future__ import division
import cPickle
import dependency_tree
import numpy as np
from collections import Counter
from scipy import stats
from scipy.stats import entropy


def get_stats(docs):
    heights = []
    node_depths = Counter()
    sentiments = []
    sent_scores = []
    other_sent_scores = []
    root_sentiments = []
    root_sent_scores = []
    root_position = Counter()
    root_first = 0
    root_last = 0
    correct_docs = 0
    num_roots = 0
    normalized_arc_lengths = []
    leaf_node_proportions = []
    parent_entropies = []

    for doc in docs:
        if doc.gold_label == doc.predicted_label:
            # how long are the edges?
            edges = doc.tree.edges
            lengths = np.zeros([len(edges)])
            for i, edge in enumerate(edges):
                lengths[i] = np.abs(edge.src_idx-edge.tgt_idx)
            lengths /= len(edges)
            normalized_arc_lengths.extend(lengths)

            # how many nodes are leaves?
            num_leaf_nodes = len(edges) - len(doc.tree.deps.keys()) + 1  # don't count the root
            leaf_node_proportions.append(num_leaf_nodes/len(edges))

            # how many children per parent?
            parents_list = []
            for i in range(1, len(edges)+1):
                parent = next(key for key, value in doc.tree.deps.items() if i in value)
                parents_list.append(parent)
            parent_entropies.append(entropy_for_parents(parents_list))

            correct_docs += 1
            heights.append(doc.tree.height)
            for key, value in doc.tree.node_depths.items():
                node_depths[key] += value
            sentiments.extend(doc.sentiments)
            sent_scores.extend(doc.sentiment_scores)
            roots = np.array(doc.tree.deps[0])-1  # need to subtract to account for 0 root in the tree

            for root in roots:
                num_roots += 1
                if root == 0:
                    root_first += 1
                elif root == len(doc.sentiments)-1:
                    root_last += 1
                # create 3 bins, for beginning, middle, end of doc
                bins = np.array_split(np.arange(len(doc.sentiments)), 3)
                for ix, bin in enumerate(bins):
                    if root in bin:
                        root_position[ix] += 1
                root_sentiments.append(doc.sentiments[root])
                root_sent_scores.append(doc.sentiment_scores[root])
            mask_roots = np.ones(len(doc.sentiment_scores), bool)
            mask_roots[roots] = False
            other_sent_scores.extend(np.array(doc.sentiment_scores)[mask_roots])

    print("Stats for normalized arc length: ", np.mean(np.array(normalized_arc_lengths)),
          np.std(np.array(normalized_arc_lengths)), np.min(np.array(normalized_arc_lengths)),
          np.max(np.array(normalized_arc_lengths)))
    print("Stats for leaf node proportions: ", np.mean(np.array(leaf_node_proportions)),
          np.std(np.array(leaf_node_proportions)), np.min(np.array(leaf_node_proportions)),
          np.max(np.array(leaf_node_proportions)))
    print("Stats for parent entropy: ", np.mean(np.array(parent_entropies)),
          np.std(np.array(parent_entropies)), np.min(np.array(parent_entropies)),
          np.max(np.array(parent_entropies)))
    print("Processed ", correct_docs, " out of ", len(docs), " documents that were labelled correctly.")
    print("\nStats for heights: ")
    print(np.mean(heights), np.std(heights), np.min(heights), np.max(heights), Counter(heights).keys(),
          Counter(heights).values() / np.sum(Counter(heights).values()))
    print("\nStats for node depths: ")
    print(node_depths.keys(), node_depths.values() / np.sum(node_depths.values()))
    print("\nStats for sentiments: ")
    print(np.mean(sentiments), np.std(sentiments), Counter(sentiments).keys(),
          Counter(sentiments).values() / np.sum(Counter(sentiments).values()))
    print("\nStats for root sentiments: ")
    print(np.mean(root_sentiments), np.std(root_sentiments), Counter(root_sentiments).keys(),
          Counter(root_sentiments).values() / np.sum(Counter(root_sentiments).values()))
    print("\nStats for sentiment scores: ")
    print(np.mean(np.abs(sent_scores)), np.std(np.abs(sent_scores)))
    print("\nStats for root sentiment scores: ")
    print(np.mean(np.abs(root_sent_scores)), np.std(np.abs(root_sent_scores)))
    print("\nRoot to position in sentence: ")
    print("First: ", root_first/num_roots, ", Last: ", root_last/num_roots, "Bins: ", root_position.keys(),
          root_position.values() / np.sum(root_position.values()))
    t_stat, p_value_two_sided = stats.ttest_ind(np.abs(root_sent_scores), np.abs(other_sent_scores), equal_var=False)
    print("T-statistic: ", t_stat, ", p_value for rejecting null hypothesis that mean(other sents) >= mean(root sents)",
          p_value_two_sided/2)


def entropy_for_parents(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


if __name__ == '__main__':
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data_structure import ProcessedDoc

    pickle_file = sys.argv[1]
    docs = cPickle.load(open(pickle_file))
    get_stats(docs)