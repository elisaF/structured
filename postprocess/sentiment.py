from itertools import groupby
import nltk
from nltk.corpus import opinion_lexicon

nltk.download('opinion_lexicon')
nltk.download('vader_lexicon')


def hu_liu_sentiment(tokenized_sent):
    pos_words = 0
    neg_words = 0
    for word in tokenized_sent:
        if word in opinion_lexicon.positive():
            pos_words += 1
        elif word in opinion_lexicon.negative():
            neg_words += 1
    if pos_words > neg_words:
        return 1
    elif pos_words < neg_words:
        return -1
    elif pos_words == neg_words:
        return 0


def vader_sentiment(tokenized_sent):
    from nltk.sentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    scores = vader_analyzer.polarity_scores(' '.join(tokenized_sent))
    return scores['compound']


def set_sentiment(docs):
    sentiments = []
    sentiment_scores = []
    for doc in docs:
        tokenized_sents = (list(g) for k,g in groupby(doc.text, key=lambda x: x != '<split>') if k)
        for sent in tokenized_sents:
            if sent:
                sentiments.append(hu_liu_sentiment(sent))
                sentiment_scores.append(vader_sentiment(sent))
        doc.set_sentiment(sentiments, sentiment_scores)
