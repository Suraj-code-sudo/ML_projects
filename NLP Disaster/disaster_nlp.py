import pandas as pd
import numpy as np
import re
import nltk

X_data = pd.read_csv("train.csv")
y_data = pd.read_csv("test.csv")

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

X_data.fillna(' ', inplace=True)
y_data.fillna(' ', inplace=True)
corpus = []
stopwords_all = stopwords.words('english')
english_words = set(nltk.corpus.words.words())
stopwords_all.remove('not')

for i in range(len(X_data)):
    tweet = X_data['keyword'][i] + " " + X_data['text'][i]
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split(' ')
    ps = SnowballStemmer(language='english')
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords_all)]
    tweet = ' '.join(tweet)
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) if w.lower() in english_words or not w.isalpha())
    corpus.append(tweet)
print(corpus[10])
corpus_test = []
for i in range(len(y_data)):
    tweet = y_data['keyword'][i] + " " + y_data['text'][i]
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split(' ')
    ps = SnowballStemmer(language='english')
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords_all)]
    tweet = ' '.join(tweet)
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) if w.lower() in english_words or not w.isalpha())
    corpus_test.append(tweet)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
X_test = cv.fit_transform(corpus_test).toarray()
y = X_data.iloc[:, -1].values

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X, y)

y_pred = classifier.predict(X_test)

y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv('results.csv')
