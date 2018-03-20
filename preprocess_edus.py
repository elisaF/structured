from os.path import join, basename
import gzip, os, sys
import fnmatch


class Document(object):
    def __init__(self, fname, doc_id, label, edu_texts):
        self.fname = fname
        self.doc_id = doc_id
        self.label = label
        self.edutexts = edu_texts


def _parse_edus(fname):
    """Take a string formatted parse (i.e., from gCRF output files)
    and extract the EDUs.
    """
    edus = []
    with open(fname, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if line.startswith("_!"):
                # Leaf / EDU
                while line.endswith(")"):
                    line = line[:-1]
                edu_text = line[2:-2]
                if edu_text.endswith("<s>") or edu_text.endswith("<P>"):
                    # self.eos = True
                    edu_text = edu_text[:-4]
                edus.append(edu_text)
    edutexts = "<split2>".join(edus)
    return edutexts


def load_labels(label_file):
    with gzip.open(label_file, 'r') as fin:
        labels = fin.read().strip().split("\n")
        print("Load {} labels from file: {}".format(len(labels), label_file))
    return labels


def parse_fname(fname):
    items = (fname.split(".")[0]).split("-")
    if len(items) != 2:
        raise ValueError("Unexpected length of items: {}".format(items))
    setlabel, fidx = items[0], int(items[1])
    return setlabel, fidx


def get_alltreefiles(tree_path, suffix="*.tree"):
    treefiles = []
    for root, dirnames, file_names in os.walk(tree_path):
        for file_name in fnmatch.filter(file_names, suffix):
            treefiles.append(join(root, file_name))
    print("Read {} files".format(len(treefiles)))
    return treefiles


def get_docdict(tree_files, trn_labels, dev_labels, tst_labels):
    trn_docdict, dev_docdict, tst_docdict = {}, {}, {}
    for tree_file in tree_files:
        fname = basename(tree_file)
        setlabel, fidx = parse_fname(fname)
        if setlabel == "train":
            doc = Document(fname, fidx, int(trn_labels[fidx]) - 1, _parse_edus(tree_file))  # convert to scale of 0-4 instead of 1-5
            trn_docdict[fname] = doc
        elif setlabel == "dev":
            doc = Document(fname, fidx, int(dev_labels[fidx]) - 1, _parse_edus(tree_file))
            dev_docdict[fname] = doc
        elif setlabel == "test":
            doc = Document(fname, fidx, int(tst_labels[fidx]) - 1, _parse_edus(tree_file))
            tst_docdict[fname] = doc
    return trn_docdict, dev_docdict, tst_docdict


def write_docs(docdict, outfname):
    print("Write docs into file: {}".format(outfname))
    with open(outfname, 'w') as fout:
        for (fname, doc) in docdict.iteritems():
            line = str(doc.label) + "<split1>" + doc.edutexts + "<split1>" + str(doc.doc_id) + "\n"
            fout.write(line)


def main():
    data_dir = sys.argv[1]
    tree_path = os.path.join(data_dir, "parses_feng/")
    trn_labels = load_labels(os.path.join(data_dir, "train.labels.gz"))
    dev_labels = load_labels(os.path.join(data_dir, "dev.labels.gz"))
    tst_labels = load_labels(os.path.join(data_dir, "test.labels.gz"))

    tree_files = get_alltreefiles(tree_path)
    trn_docdict, dev_docdict, tst_docdict = get_docdict(tree_files, trn_labels, dev_labels, tst_labels)
    f_trn = os.path.join(data_dir, "processed_yelp_sentiment_edus_train.txt")
    f_dev = os.path.join(data_dir, "processed_yelp_sentiment_edus_dev.txt")
    f_tst = os.path.join(data_dir, "processed_yelp_sentiment_edus_test.txt")
    write_docs(trn_docdict, f_trn)
    write_docs(dev_docdict, f_dev)
    write_docs(tst_docdict, f_tst)


if __name__ == '__main__':
    main()
