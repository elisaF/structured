from os.path import join
import gzip, os, sys
import fnmatch


def load_labels(label_file):
    with gzip.open(label_file, 'r') as fin:
        labels = fin.read().strip().split("\n")
        print("Load {} labels from file: {}".format(len(labels), label_file))
    return labels


def get_allbracketsfiles(rpath, suffix="*.brackets"):
    bracketsfiles = []
    for root, dirnames, file_names in os.walk(rpath):
        for file_name in fnmatch.filter(file_names, suffix):
            bracketsfiles.append(join(root, file_name))
    print("Read {} files".format(len(bracketsfiles)))
    return bracketsfiles


class Document(object):
    def __init__(self):
        self.edutexts = ""
        self.label = -1
        self.doc_id = -1
        self.fname = None

    def _load_segmentation(self):
        """ Load discourse segmentation results
        """
        edus = {}
        with open(self.fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split("\t")
                token, idx = None, int(items[-1])
                token = items[2]
                try:
                    edus[idx] += (" " + token)
                except KeyError:
                    edus[idx] = token
        self.edutexts = "<split2>".join(edus.values())

def main():
    data_dir = sys.argv[1]
    rpath = os.path.join(data_dir, "feng_parses/")
    trn_labels = load_labels(os.path.join(data_dir, "train.labels.gz"))
    dev_labels = load_labels(os.path.join(data_dir, "dev.labels.gz"))
    tst_labels = load_labels(os.path.join(data_dir, "test.labels.gz"))

