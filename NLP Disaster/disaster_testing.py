import pandas as pd
import numpy as np
import re
import nltk

X_data = pd.read_csv("train.csv")
y_data = pd.read_csv("test.csv")

y = X_data.iloc[: ,-1].values
print(y)
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

X_data.fillna(' ', inplace=True)
y_data.fillna(' ', inplace=True)
corpus = []
stopwords_all = stopwords.words('english')
stopwords_all.remove('not')
english_words = set(nltk.corpus.words.words())


for i in range(len(X_data)):
    tweet = X_data['keyword'][i] + " " + X_data['text'][i]
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split(' ')
    ps = SnowballStemmer(language='english')
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords_all)]
    tweet = " ".join(tweet)
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) if w.lower() in english_words or not w.isalpha())
    corpus.append(tweet)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv('results.csv')

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))