"""
Basic example of training a model for Twitter sentiment analysis.

Utilises the Sanders Twitter Sentiment Corpus.

http://www.sananalytics.com/lab/twitter-sentiment/
"""

import pandas
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

twitter_data = pandas.read_csv(
    open('./twitter-sentiment-dataset/full-corpus.csv'))
twitter_data = twitter_data[twitter_data['Sentiment'] != 'neutral']
twitter_data = twitter_data[twitter_data['Sentiment'] != 'irrelevant']
twitter_data['target'] = twitter_data['Sentiment'].apply(lambda x: +1 if x == 'positive' else -1) 

train, test = train_test_split(twitter_data, train_size=0.7)

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())])

text_clf = text_clf.fit(train['TweetText'], train['target'])

predicted = text_clf.predict(test['TweetText'])
print np.mean(predicted == test['target'])
