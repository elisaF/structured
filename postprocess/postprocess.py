import cPickle
import postprocess.sentiment
import sys


if __name__ == '__main__':
    pickle_file = sys.argv[1]
    docs = cPickle.load(open(pickle_file))
    postprocess.sentiment.set_sentiment(docs)
    pickle_file_updated = pickle_file + "_withsent"
    cPickle.dump(docs, open(pickle_file_updated, 'w'))
