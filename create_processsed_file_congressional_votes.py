import os, sys


def _get_file_names(dir_, suffix=".txt"):
    return [f for f in os.listdir(dir_) if f.endswith(suffix)]


def create_file(data_dir, split):
    text_blobs = []
    files = _get_file_names(data_dir)
    for file in files:
        items = file.split(".")[0].split("_")
        doc_id = items[0] + items[1]
        doc_label = "0" if items[2] == "N" else "1"
        with open(os.path.join(data_dir, file)) as f:
            sents = f.readlines()
            clean_sents = [sent.rstrip("\n") for sent in sents]
            text_blob = "<split2>".join(clean_sents)
        text_blob = doc_label + "<split1>" + text_blob + "<split1>" + doc_id
        text_blobs.append(text_blob)
    parent_dir = os.path.abspath(os.path.join(data_dir, os.pardir))
    processed_file_name = os.path.join(parent_dir, "processed_congressional_votes_" + split + ".txt")
    with open(processed_file_name, 'w') as f:
        for text_blob in text_blobs:
            f.write("{}\n".format(text_blob))


if __name__ == '__main__':
    data_dir = sys.argv[1]

    dev_split = "development"
    dev_dir = os.path.join(data_dir, dev_split + "_set")
    test_split = "test"
    test_dir = os.path.join(data_dir, test_split + "_set")
    train_split = "training"
    train_dir = os.path.join(data_dir, train_split + "_set")

    create_file(dev_dir, dev_split)
    create_file(test_dir, test_split)
    create_file(train_dir, train_split)
