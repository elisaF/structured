# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
import csv
import tarfile
import os, sys
from bs4 import BeautifulSoup
import numpy as np
from random import shuffle

labels = ["typical", "verygood", "great"]


def _get_file_names(dir_, suffix=".txt"):
    return [os.path.join(dir_, f) for f in os.listdir(dir_) if f.endswith(suffix)]


def make_dict():
    return defaultdict(make_dict)


def create_data_splits(data_file_dir, paired_file_list, trn_perc=.8, tst_perc=.1, dev_perc=.1):
    # create two classes
    verygood_list = []
    typical_list = []
    data_files = _get_file_names(data_file_dir, ".txt")
    for data_file in data_files:
        label = os.path.basename(data_file).split(".")[0]
        if label == labels[0]:
            with open(data_file) as f:
                typical_list.extend([line.rstrip("\n") for line in f.readlines()])
        elif label == labels[1] or label == labels[2]:
            with open(data_file) as f:
                verygood_list.extend([line.rstrip("\n") for line in f.readlines()])
        else:
            print("Ignoring file: ", data_file)
    shuffle(typical_list)
    shuffle(verygood_list)

    # use smaller class to determine size of data splits
    trn_num = int(np.floor(len(verygood_list) * trn_perc)) + 1  # TODO: Fix this hack!
    tst_num = int(np.floor(len(verygood_list) * tst_perc))
    dev_num = int(np.floor(len(verygood_list) * dev_perc))

    verygood_trn = verygood_list[:trn_num]
    verygood_tst = verygood_list[trn_num:trn_num + tst_num]
    verygood_dev = verygood_list[trn_num + tst_num:trn_num + tst_num + dev_num]

    # get paired typical files with similar topic
    typical_trn, typical_tst, typical_dev = get_paired_files(verygood_trn, verygood_tst, verygood_dev, paired_file_list, typical_list)

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


def get_paired_files(verygood_trn, verygood_tst, verygood_dev, paired_file_list, typical_list):
    paired_files_trn = []
    paired_files_tst = []
    paired_files_dev = []

    num_not_topic_controlled_trn = 0
    num_not_topic_controlled_tst = 0
    num_not_topic_controlled_dev = 0

    paired_dict = create_paired_dict(paired_file_list)
    #all_matched_files = set([item for sublist in paired_dict.values() for item in sublist])
    typical_list = set(typical_list)
    for file in verygood_trn:
        matches = paired_dict[file]
        if matches[0] not in paired_files_trn and matches[0] not in paired_files_tst and matches[0] not in paired_files_dev:
            paired_files_trn.append(matches[0])
            typical_list.remove(matches[0])
        elif matches[1] not in paired_files_trn and matches[1] not in paired_files_tst and matches[1] not in paired_files_dev:
            paired_files_trn.append(matches[1])
            typical_list.remove(matches[1])
        elif matches[2] not in paired_files_trn and matches[2] not in paired_files_tst and matches[2] not in paired_files_dev:
            paired_files_trn.append(matches[2])
            typical_list.remove(matches[2])
        elif matches[3] not in paired_files_trn and matches[3] not in paired_files_tst and matches[3] not in paired_files_dev:
            paired_files_trn.append(matches[3])
            typical_list.remove(matches[3])
        elif matches[4] not in paired_files_trn and matches[4] not in paired_files_tst and matches[4] not in paired_files_dev:
            paired_files_trn.append(matches[4])
            typical_list.remove(matches[4])
        elif matches[5] not in paired_files_trn and matches[5] not in paired_files_tst and matches[5] not in paired_files_dev:
            paired_files_trn.append(matches[5])
            typical_list.remove(matches[5])
        elif matches[6] not in paired_files_trn and matches[6] not in paired_files_tst and matches[6] not in paired_files_dev:
            paired_files_trn.append(matches[6])
            typical_list.remove(matches[6])
        elif matches[7] not in paired_files_trn and matches[7] not in paired_files_tst and matches[7] not in paired_files_dev:
            paired_files_trn.append(matches[7])
            typical_list.remove(matches[7])
        elif matches[8] not in paired_files_trn and matches[8] not in paired_files_tst and matches[8] not in paired_files_dev:
            paired_files_trn.append(matches[8])
            typical_list.remove(matches[8])
        elif matches[9] not in paired_files_trn and matches[9] not in paired_files_tst and matches[9] not in paired_files_dev:
            paired_files_trn.append(matches[9])
            typical_list.remove(matches[9])
        else:
            #print("TRAIN: All paired files for very good file: ", file, " have been included in the typical list. Going to include one that is not topic-controlled.")
            paired_files_trn.append(typical_list.pop())
            num_not_topic_controlled_trn += 1

    for file in verygood_tst:
        matches = paired_dict[file]
        if matches[0] not in paired_files_trn and matches[0] not in paired_files_tst and matches[0] not in paired_files_dev:
            paired_files_tst.append(matches[0])
            typical_list.remove(matches[0])
        elif matches[1] not in paired_files_trn and matches[1] not in paired_files_tst and matches[1] not in paired_files_dev:
            paired_files_tst.append(matches[1])
            typical_list.remove(matches[1])
        elif matches[2] not in paired_files_trn and matches[2] not in paired_files_tst and matches[2] not in paired_files_dev:
            paired_files_tst.append(matches[2])
            typical_list.remove(matches[2])
        elif matches[3] not in paired_files_trn and matches[3] not in paired_files_tst and matches[3] not in paired_files_dev:
            paired_files_tst.append(matches[3])
            typical_list.remove(matches[3])
        elif matches[4] not in paired_files_trn and matches[4] not in paired_files_tst and matches[4] not in paired_files_dev:
            paired_files_tst.append(matches[4])
            typical_list.remove(matches[4])
        elif matches[5] not in paired_files_trn and matches[5] not in paired_files_tst and matches[5] not in paired_files_dev:
            paired_files_tst.append(matches[5])
            typical_list.remove(matches[5])
        elif matches[6] not in paired_files_trn and matches[6] not in paired_files_tst and matches[6] not in paired_files_dev:
            paired_files_tst.append(matches[6])
            typical_list.remove(matches[6])
        elif matches[7] not in paired_files_trn and matches[7] not in paired_files_tst and matches[7] not in paired_files_dev:
            paired_files_tst.append(matches[7])
            typical_list.remove(matches[7])
        elif matches[8] not in paired_files_trn and matches[8] not in paired_files_tst and matches[8] not in paired_files_dev:
            paired_files_tst.append(matches[8])
            typical_list.remove(matches[8])
        elif matches[9] not in paired_files_trn and matches[9] not in paired_files_tst and matches[9] not in paired_files_dev:
            paired_files_tst.append(matches[9])
            typical_list.remove(matches[9])
        else:
            #print("TEST: All paired files for very good file: ", file, " have been included in the typical list. Going to include one that is not topic-controlled.")
            paired_files_tst.append(typical_list.pop())
            num_not_topic_controlled_tst += 1

    for file in verygood_dev:
        matches = paired_dict[file]
        if matches[0] not in paired_files_trn and matches[0] not in paired_files_tst and matches[0] not in paired_files_dev:
            paired_files_dev.append(matches[0])
            typical_list.remove(matches[0])
        elif matches[1] not in paired_files_trn and matches[1] not in paired_files_tst and matches[1] not in paired_files_dev:
            paired_files_dev.append(matches[1])
            typical_list.remove(matches[1])
        elif matches[2] not in paired_files_trn and matches[2] not in paired_files_tst and matches[2] not in paired_files_dev:
            paired_files_dev.append(matches[2])
            typical_list.remove(matches[2])
        elif matches[3] not in paired_files_trn and matches[3] not in paired_files_tst and matches[3] not in paired_files_dev:
            paired_files_dev.append(matches[3])
            typical_list.remove(matches[3])
        elif matches[4] not in paired_files_trn and matches[4] not in paired_files_tst and matches[4] not in paired_files_dev:
            paired_files_dev.append(matches[4])
            typical_list.remove(matches[4])
        elif matches[5] not in paired_files_trn and matches[5] not in paired_files_tst and matches[5] not in paired_files_dev:
            paired_files_dev.append(matches[5])
            typical_list.remove(matches[5])
        elif matches[6] not in paired_files_trn and matches[6] not in paired_files_tst and matches[6] not in paired_files_dev:
            paired_files_dev.append(matches[6])
            typical_list.remove(matches[6])
        elif matches[7] not in paired_files_trn and matches[7] not in paired_files_tst and matches[7] not in paired_files_dev:
            paired_files_dev.append(matches[7])
            typical_list.remove(matches[7])
        elif matches[8] not in paired_files_trn and matches[8] not in paired_files_tst and matches[8] not in paired_files_dev:
            paired_files_dev.append(matches[8])
            typical_list.remove(matches[8])
        elif matches[9] not in paired_files_trn and matches[9] not in paired_files_tst and matches[9] not in paired_files_dev:
            paired_files_dev.append(matches[9])
            typical_list.remove(matches[9])
        else:
            #print("DEV: All paired files for very good file: ", file, " have been included in the typical list. Going to include one that is not topic-controlled.")
            paired_files_dev.append(typical_list.pop())
            num_not_topic_controlled_dev += 1

    print("Total number of files that don't control for topic: (trn/tst/dev)", num_not_topic_controlled_trn, num_not_topic_controlled_tst, num_not_topic_controlled_dev)
    return paired_files_trn, paired_files_tst, paired_files_dev


def create_paired_dict(paired_list_file):
    paired_dict = {}
    with open(paired_list_file) as f:
        for line in f.readlines():
            tabbed_line = line.split("\t")
            file_to_match = tabbed_line[0]
            matches = []
            for index in range(2,12):
                matches.append(tabbed_line[index].split(":")[0])
            paired_dict[file_to_match] = matches
    return paired_dict


def create_corpus(data_file_dir, paired_file_list, ldc_tar_file, new_corpus_dir):
    data_splits = create_data_splits(data_file_dir, paired_file_list)
    write_files(data_splits, ldc_tar_file, new_corpus_dir)


def write_files(data_splits, ldc_tar_file, new_corpus_dir):
    dir_to_xml = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    xml_to_data_class_split = {}
    # for split in data_splits:
    #     for label in split:
    #         print("Split ", split, " for class ", label, " has length: ", len(label), ", set: ", len(set(label)))


    # create mappings of file to directory location
    for split_idx, data_split in enumerate(data_splits):
        for label_idx, label_data_split in enumerate(data_split):
            for file_name in label_data_split:
                year, folder1, folder2, xml = file_name.split("_")
                dir_to_xml[year][folder1][folder2].append(xml)
                assert (xml not in xml_to_data_class_split), "XML was found!"+xml  # sanity check to make sure xmls are unique
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
                            writer.writerow(["".join([year, folder1, folder2, xml.split(".")[0]]), plain_text.encode('utf-8'), label_idx])
                            csv_counter[split_idx] += 1

    print("Finished writing corpora: ", csv_counter)
    print("Total number of xmls written: ", len(set(all_xmls)))


def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


if __name__ == '__main__':
    data_file_dir = sys.argv[1]
    paired_file_list = sys.argv[2]
    ldc_tar_file = sys.argv[3]
    new_corpus_dir = sys.argv[4]

    create_corpus(data_file_dir, paired_file_list, ldc_tar_file, new_corpus_dir)
