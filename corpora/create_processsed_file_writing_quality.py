# -*- coding: utf-8 -*-
import csv
import os, sys

def create_files(data_dir):


def create_file(csv_file, split):
    text_blobs = []
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            doc_id = row[0]
            text_blob = "<split2>".join(row[1].decode('utf-8'))
            doc_label = row[2]
            text_blob = doc_label + "<split1>" + text_blob + "<split1>" + doc_id
            text_blobs.append(text_blob)
    parent_dir = os.path.abspath(os.path.join(data_dir, os.pardir))
    processed_file_name = os.path.join(parent_dir, "processed_writing_quality_" + split + ".txt")
    with open(processed_file_name, 'w') as f:
        for text_blob in text_blobs:
            f.write("{}\n".format(text_blob))


if __name__ == '__main__':

    data_file = sys.argv[1]
data = {}
with open(data_file) as f:
    for line in f:
        doc_id, bill_json=line.split('\t')
        bill_text=json.loads(bill_json)
        try:
            bill_data[doc_id] = bill_text['text']
        except KeyError:
            print("Can't get text for: ", doc_id)
    dev_split = "development"
    dev_dir = os.path.join(data_dir, dev_split + "_set")
    test_split = "test"
    test_dir = os.path.join(data_dir, test_split + "_set")
    train_split = "training"
    train_dir = os.path.join(data_dir, train_split + "_set")

    create_file(dev_dir, dev_split)
    create_file(test_dir, test_split)
    create_file(train_dir, train_split)
