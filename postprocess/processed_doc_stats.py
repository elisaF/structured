from __future__ import division
import cPickle
import dependency_tree
import numpy as np
from collections import Counter


def get_stats(docs):
    heights = []
    node_depths = Counter()
    sentiments = []
    sent_scores = []
    root_sentiments = []
    root_sent_scores = []
    root_position = Counter()
    root_first = 0
    root_last = 0
    correct_docs = 0
    num_roots = 0
    for doc in docs:
        if doc.gold_label == doc.predicted_label:
            correct_docs += 1
            heights.append(doc.tree.height)
            for key, value in doc.tree.node_depths.items():
                node_depths[key] += value
            sentiments.extend(doc.sentiments)
            sent_scores.extend(doc.sentiment_scores)
            for root in doc.tree.deps[0]:
                num_roots += 1
                if root == 1:
                    root_first += 1
                elif root == len(doc.sentiments):
                    root_last += 1
                # create 3 bins, for beginning, middle, end of doc
                bins = np.array_split(np.arange(len(doc.sentiments)), 3)
                for ix, bin in enumerate(bins):
                    if root in bin:
                        root_position[ix] += 1
                root_sentiments.append(doc.sentiments[root-1])  # need to subtract to account for 0 root in the tree
                root_sent_scores.append(doc.sentiment_scores[root - 1])

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


if __name__ == '__main__':
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data_structure import ProcessedDoc

    pickle_file = sys.argv[1]
    docs = cPickle.load(open(pickle_file))
    get_stats(docs)