import json
import numpy as np
from collections import Counter
import csv
import os, sys
from sklearn.model_selection import train_test_split


def create_corpus(json_file, save_path):
    X_ids_texts, y_frames = read_corpus(json_file)
    create_partitions(X_ids_texts, y_frames, save_path)


def read_corpus(json_file):
    irrelevant_count = 0
    X_ids_texts = []
    y_frames = []
    docs = json.load(open(json_file))
    for id in docs:
        id_num = id.split("-")[1]
        frame = docs[id]['primary_frame']
        tone = docs[id]['primary_tone']
        irrelevant = docs[id]['irrelevant']
        if frame and irrelevant:
            irrelevant_count += 1
        if frame and tone and not irrelevant:
            frame = str(frame).split(".")[0]
            y_frames.append(frame)
            text = [par for par in docs[id]['text'].split("\n") if len(par)>0][2:]
            X_ids_texts.append((id_num, text))
    print("Class frequencies: ", Counter(y_frames))
    print("Irrelevant frames: ", irrelevant_count)
    return X_ids_texts, y_frames


def create_partitions(X_data, y_data, save_path):
    X_train, X_devtest, y_train, y_devtest = train_test_split(X_data, y_data,
                                                    random_state=4,
                                                    stratify=y_data,
                                                    test_size=0.2)

    X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest,
                                                    random_state=4,
                                                    stratify=y_devtest,
                                                    test_size=0.5)

    # Sanity checks
    dev_ids = []
    for id, _ in X_dev:
        dev_ids.append(id)
    dev_ids = set(dev_ids)

    train_ids = []
    for id, _ in X_train:
        train_ids.append(id)
    train_ids = set(train_ids)
    train_ids = set(train_ids)

    test_ids = []
    for id, _ in X_test:
        test_ids.append(id)
    test_ids = set(test_ids)
    test_ids = set(test_ids)

    assert(len(train_ids.intersection(dev_ids))==0)
    assert(len(train_ids.intersection(test_ids))==0)
    assert(len(test_ids.intersection(dev_ids))==0)

    Xs = [X_train, X_test, X_dev]
    ys = [y_train, y_test, y_dev]

    write_partitions(Xs, ys, save_path)

    print("Class frequencies for train / dev / test: ", Counter(y_train), Counter(y_dev), Counter(y_test))


def write_partitions(Xs, ys, save_path):
    csv_counter = Counter()

    trn_file = os.path.join(save_path, "mfc_trn.csv")
    tst_file = os.path.join(save_path, "mfc_tst.csv")
    dev_file = os.path.join(save_path, "mfc_dev.csv")
    file_names = [trn_file, tst_file, dev_file]

    for split_index in range(3):
        with open(file_names[split_index], 'a+') as fout:
            for index, (id, text) in enumerate(Xs[split_index]):
                frame = ys[split_index][index]
                writer = csv.writer(fout)
                writer.writerow([id, ' '.join(text).encode('utf-8'), frame])
                csv_counter[split_index] += 1
    print("Finished writing corpora: ", csv_counter)

if __name__ == '__main__':
    json_file = sys.argv[1]
    save_path = sys.argv[2]
    create_corpus(json_file, save_path)