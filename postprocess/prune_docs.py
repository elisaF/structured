import cPickle
import numpy as np
import pandas as pd
import re

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data_structure import ProcessedDoc

    pickle_file = sys.argv[1]
    text_file = sys.argv[2]

    text_file_updated = text_file + "_pruned"
    docs = cPickle.load(open(pickle_file))
    column_names = ["Label", "Text", "Doc_id"]
    df = pd.read_csv(text_file, "<split1>", encoding='utf-8', names=column_names)
    new_rows = []
    for doc in docs:
        roots = np.array(doc.tree.deps[0])
        processed_roots = np.array(' '.join(doc.text).split("<split>"))[roots - 1]
        preprocessed_sents = np.array(df.loc[df.Doc_id==doc.doc_id].Text.values[0].split("<split2>"))
        # do the same processing as before, to ensure indices align
        preprocessed_sents = [re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ", sent) for sent in preprocessed_sents]
        preprocessed_sents = np.array([sent for sent in preprocessed_sents if len(sent.rstrip(" ")) > 0])
        preprocessed_roots = preprocessed_sents[roots - 1]
        # sanity check
        for i in range(len(roots)):
            # make sure there is a reasonable amount of overlap in tokens between texts
            root_words = set([word for word in processed_roots[i].split(" ") if len(word.rstrip(" ")) > 0])
            preprocessed_words = set([word for word in preprocessed_roots[i].split(" ") if len(word.rstrip(" ")) > 0])
            required_overlap = len(root_words) * .75
            actual_overlap = len(preprocessed_words.intersection(root_words))
            if (actual_overlap < required_overlap):
                print("Mismatch in texts for doc id ", doc.doc_id, ": ", processed_roots[i], preprocessed_roots[i])
        new_sents = "<split2>".join(preprocessed_roots)
        new_rows.append("<split1>".join([str(doc.gold_label), new_sents, str(doc.doc_id)]))

    with open(text_file_updated, 'w') as fout:
        fout.write('\n'.join(new_rows))

    print("Processed ", len(docs), " in pickle file ", pickle_file, " into ", len(new_rows),
          " lines using preprocessed text from ", text_file)
