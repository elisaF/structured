import cPickle
import dependency_tree
import sentiment

if __name__ == '__main__':
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data_structure import ProcessedDoc

    pickle_file = sys.argv[1]
    docs = cPickle.load(open(pickle_file))
    sentiment.calculate_sentiment(docs)
    pickle_file_updated = pickle_file + "_withtreesent"
    dependency_tree.calculate_tree(docs)
    cPickle.dump(docs, open(pickle_file_updated, 'w'))
