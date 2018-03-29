from itertools import groupby
import nltk
from nltk.corpus import opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer
import datetime

nltk.download('opinion_lexicon')
nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()
set_pos_words = set(opinion_lexicon.positive())
set_neg_words = set(opinion_lexicon.negative())


def hu_liu_sentiment(tokenized_sent):
    pos_words = 0
    neg_words = 0
    for word in tokenized_sent:
        word = word.lower()
        if word in set_pos_words:
            pos_words += 1
        elif word in set_neg_words:
            neg_words += 1
    if pos_words > neg_words:
        return 1
    elif pos_words < neg_words:
        return -1
    elif pos_words == neg_words:
        return 0


def vader_sentiment(tokenized_sent):
    scores = vader_analyzer.polarity_scores(' '.join(tokenized_sent))
    return scores['compound']


def calculate_sentiment(docs):
    for doc_counter, doc in enumerate(docs):
        sentiments = []
        sentiment_scores = []
        tokenized_sents = (list(g) for k,g in groupby(doc.text, key=lambda x: x != '<split>') if k)
        for sent in tokenized_sents:
            if sent:
                sentiments.append(hu_liu_sentiment(sent))
                sentiment_scores.append(vader_sentiment(sent))
        doc.set_sentiment(sentiments, sentiment_scores)
        if doc_counter % 200 == 0:
            print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            print("Processed sentiment for ", str(doc_counter+1), " documents.")

