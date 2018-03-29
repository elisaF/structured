import tarfile
import os, sys
from bs4 import BeautifulSoup

if __name__ == '__main__':
    data_file = sys.argv[1]
    ldc_tar_file = sys.argv[2]
    corpus_files = {}

    with open(data_file) as f:
        for line in f:
            year, folder1, folder2, xml = line.split("_")
            corpus_files[year] = [folder1, folder2]

    with tarfile.open(ldc_tar_file, "r") as tar:
        bar = tarfile.open(fileobj=tar.extractfile("nyt_corpus/data/2003/01.tgz"))
        xml_text = bar.extractfile('01/01/1453064.xml').read()
        soup = BeautifulSoup(xml_text, "lxml")
        assert(len(soup.find_all("block", class_="full_text")) == 1)  # sanity check
        plain_text = soup.find_all("block", class_="full_text")[0].text


