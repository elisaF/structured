# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
import csv
import tarfile
import os, sys
from bs4 import BeautifulSoup
import numpy as np
from random import shuffle

labels = ["typical", "verygood"]


def _get_file_names(dir_, suffix=".txt"):
    return [os.path.join(dir_, f) for f in os.listdir(dir_) if f.endswith(suffix)]


def make_dict():
    return defaultdict(make_dict)


def create_data_splits(data_file_dir, trn_perc=.8, tst_perc=.1, dev_perc=.1):
    # create two classes
    verygood_list = []
    typical_list = []
    data_files = _get_file_names(data_file_dir, ".txt")
    for data_file in data_files:
        data_class = labels[0] if os.path.basename(data_file).split(".")[0] == labels[0] else labels[1]
        if data_class == labels[0]:
            with open(data_file) as f:
                typical_list.extend([line.rstrip("\n") for line in f.readlines()])
        else:
            with open(data_file) as f:
                verygood_list.extend([line.rstrip("\n") for line in f.readlines()])

    # shuffle(typical_list)
    # shuffle(verygood_list)

    # use smaller class to determine size of data splits
    trn_num = int(np.floor(len(verygood_list) * trn_perc)) + 1  # TODO: Fix this hack!
    tst_num = int(np.floor(len(verygood_list) * tst_perc))
    dev_num = int(np.floor(len(verygood_list) * dev_perc))

    # split data sets
    typical_trn = typical_list[:trn_num]
    typical_tst = typical_list[trn_num:trn_num + tst_num]
    typical_dev = typical_list[trn_num + tst_num:trn_num + tst_num + dev_num]

    verygood_trn = verygood_list[:trn_num]
    verygood_tst = verygood_list[trn_num:trn_num + tst_num]
    verygood_dev = verygood_list[trn_num + tst_num:trn_num + tst_num + dev_num]

    # write to files for reference
    with open(os.path.join(data_file_dir, labels[0] + "_trn.list"), 'w') as f:
        f.write("\n".join(typical_trn))
    with open(os.path.join(data_file_dir, labels[0] + "_tst.list"), 'w') as f:
        f.write("\n".join(typical_tst))
    with open(os.path.join(data_file_dir, labels[0] + "_dev.list"), 'w') as f:
        f.write("\n".join(typical_dev))

    with open(os.path.join(data_file_dir, labels[1] + "_trn.list"), 'w') as f:
        f.write("\n".join(verygood_trn))
    with open(os.path.join(data_file_dir, labels[1] + "_tst.list"), 'w') as f:
        f.write("\n".join(verygood_tst))
    with open(os.path.join(data_file_dir, labels[1] + "_dev.list"), 'w') as f:
        f.write("\n".join(verygood_dev))

    print("Wrote indeces files.")

    return ([typical_trn, verygood_trn],
            [typical_tst, verygood_tst],
            [typical_dev, verygood_dev])


def create_corpus(data_file_dir, ldc_tar_file, new_corpus_dir):
    data_splits = create_data_splits(data_file_dir)
    write_files(data_splits, ldc_tar_file, new_corpus_dir)


def write_files(data_splits, ldc_tar_file, new_corpus_dir):
    dir_to_xml = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    xml_to_data_class_split = {}

    # create mappings of file to directory location
    for split_idx, data_split in enumerate(data_splits):
        for label_idx, label_data_split in enumerate(data_split):
            for file_name in label_data_split:
                year, folder1, folder2, xml = file_name.split("_")
                dir_to_xml[year][folder1][folder2].append(xml)
                assert (xml not in xml_to_data_class_split)  # sanity check to make sure xmls are unique
                xml_to_data_class_split[xml] = [split_idx, label_idx]
    print("Done creating mapping.")

    # write files to new corpus dir
    make_dir(new_corpus_dir)
    trn_file = os.path.join(new_corpus_dir, "writing_quality_trn.csv")
    tst_file = os.path.join(new_corpus_dir, "writing_quality_tst.csv")
    dev_file = os.path.join(new_corpus_dir, "writing_quality_dev.csv")
    file_names = [trn_file, tst_file, dev_file]
    csv_counter = Counter()
    all_xmls = []

    with tarfile.open(ldc_tar_file, "r") as tar:
        for year in dir_to_xml.keys():
            for folder1 in dir_to_xml[year]:
                folder1_unzipped = tarfile.open(
                    fileobj=tar.extractfile(os.path.join("nyt_corpus", "data", year, folder1 + ".tgz")))
                for folder2 in dir_to_xml[year][folder1]:
                    for xml in dir_to_xml[year][folder1][folder2]:
                        xml_text = folder1_unzipped.extractfile(os.path.join(folder1, folder2, xml)).read()
                        soup = BeautifulSoup(xml_text, "lxml")
                        assert (len(soup.find_all("block", class_="full_text")) == 1)  # sanity check
                        plain_text = soup.find_all("block", class_="full_text")[0].text
                        split_idx, label_idx = xml_to_data_class_split[xml]
                        all_xmls.append(xml)
                        with open(file_names[split_idx], 'a+') as fout:
                            writer = csv.writer(fout)
                            writer.writerow(["_".join([year, folder1, folder2, xml]), plain_text.encode('utf-8'), label_idx])
                            csv_counter[split_idx] += 1

    print("Finished writing corpora: ", csv_counter)
    print("Total number of xmls written: ", len(set(all_xmls)))

def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


if __name__ == '__main__':
    data_file_dir = sys.argv[1]
    ldc_tar_file = sys.argv[2]
    new_corpus_dir = sys.argv[3]

    create_corpus(data_file_dir, ldc_tar_file, new_corpus_dir)
