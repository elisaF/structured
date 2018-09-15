import os, sys


def add_id(old_file, new_file):
    updated_docs = []
    with open(old_file) as f:
        docs = f.readlines()
        for idx, doc in enumerate(docs):
            updated_docs.append(doc.rstrip("\n") + "<split1>" + str(idx)+"\n")

    with open(new_file, 'w') as f:
        f.writelines(updated_docs)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    train_file = sys.argv[2]
    dev_file = sys.argv[3]
    test_file = sys.argv[4]

    add_id(os.path.join(data_dir, train_file), os.path.join(data_dir, "with_id_"+train_file))
    add_id(os.path.join(data_dir, dev_file), os.path.join(data_dir, "with_id_" + dev_file))
    add_id(os.path.join(data_dir, test_file), os.path.join(data_dir, "with_id_" + test_file))
