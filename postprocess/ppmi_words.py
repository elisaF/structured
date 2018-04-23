from __future__ import division
from __future__ import unicode_literals
import cPickle
import dependency_tree
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from spacy.en import English
parser = English()


def tokenizeText(sentence):
    sentence = sentence.decode('utf8')
    # get the tokens using spaCy
    tokens = parser(sentence)
    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    return tokens


def get_words(docs):
    root_words = []
    other_words = []
    for doc in docs:
        sentences = np.array([sent for sent in ' '.join(doc.text).split("<split>") if len(sent.rstrip()) > 0])
        roots = np.array(doc.tree.deps[0]) - 1  # need to subtract to account for 0 root in the tree
        root_sentences = sentences[roots]
        root_sentences_tokenized = [tokenizeText(sent) for sent in root_sentences]
        root_words.extend([word for sent in root_sentences_tokenized for word in sent])
        mask_roots = np.ones(len(sentences), bool)
        mask_roots[roots] = False
        other_sentences = sentences[mask_roots]
        other_sentences_tokenized = [tokenizeText(sent) for sent in other_sentences]
        other_words.extend([word for sent in other_sentences_tokenized for word in sent])
    vocab = set(root_words).union(set(other_words))
    print("Root words: ", len(root_words), ", other words: ", len(other_words), ", vocabulary: ", len(vocab))
    return root_words, other_words


def get_word_counts_and_index(root_words, other_words, min_freq):
    vocab_vectorizer = CountVectorizer(ngram_range=(1, 1), tokenizer=tokenizeText, lowercase=False,
                                 token_pattern='\b*', min_df=min_freq)
    vocab_vectorizer.fit_transform(root_words+other_words)
    freq_vocab = vocab_vectorizer.vocabulary_.keys()
    vectorizer = CountVectorizer(ngram_range=(1, 1), tokenizer=tokenizeText, vocabulary=freq_vocab, lowercase=False,
                                 token_pattern='\b*')
    root_counts = vectorizer.fit_transform(root_words)
    other_counts = vectorizer.fit_transform(other_words)
    print("Got frequency counts for root: ", root_counts.shape, ", other: ", other_counts.shape,
          " with vocab: ", len(vectorizer.vocabulary_))
    return root_counts, other_counts, vectorizer.vocabulary_


def calculate_ppmi(root_counts, other_counts):
    count_root = root_counts.sum(axis=0)  # 1 x vocab_len matrix
    count_other = other_counts.sum(axis=0)
    count_all = np.sum(count_root + count_other)

    prob_word_root = count_root / count_all
    prob_word = (count_root + count_other) / count_all
    prob_root = np.sum(count_root) / count_all

    # divide by prob target
    pmi_target_context_matrix = prob_word_root / prob_word

    # divide by prob context
    pmi_target_context_matrix = pmi_target_context_matrix / prob_root

    # take log -- this will generate a divide by zero warning because we are taking log of 0
    pmi_target_context_matrix = np.log(pmi_target_context_matrix)
    ppmi_target_context_matrix = np.maximum(pmi_target_context_matrix, 0)
    print("Calculated ppmi matrix: ", ppmi_target_context_matrix.shape)
    return ppmi_target_context_matrix


def get_highest_ppmi(docs, save_file, min_freq, top_k=20):
    root_words, other_words = get_words(docs)
    root_counts, other_counts, word_to_index = get_word_counts_and_index(root_words, other_words, min_freq)
    target_context_matrix_ppmi = calculate_ppmi(root_counts, other_counts)
    sorted_ppmi = np.array(np.argsort(target_context_matrix_ppmi)).reshape(-1, )  # from lowest to highest
    sorted_ppmi = sorted_ppmi[::-1]  # reverse, from highest to lowest
    sorted_words = [lookup_value(word_to_index, index) for index in sorted_ppmi]
    with open(save_file, 'wb') as fn:
        cPickle.dump(sorted_words, fn)
    top_k_words = sorted_words[:top_k]
    return top_k_words


def lookup_value(_dict, _value):
    return next(key for key, value in _dict.items() if value == _value)


if __name__ == '__main__':
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data_structure import ProcessedDoc

    pickle_file = sys.argv[1]
    save_file = sys.argv[2]
    min_freq = int(sys.argv[3])
    docs = cPickle.load(open(pickle_file))
    print("Minimum frequency is: ", min_freq)
    top_words = get_highest_ppmi(docs, save_file, min_freq)
    print("Top words for ", pickle_file, " are: ", top_words)