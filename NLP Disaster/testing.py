import pandas as pd
import numpy as np
import re
import nltk
nltk.download('words')
X_data = pd.read_csv("train.csv")
y_data = pd.read_csv("test.csv")

y = X_data.iloc[: ,-1].values
print(y)
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

english_words = set(nltk.corpus.words.words())

sent = "Io andiamo to the beach with my amico."

res = " ".join(w for w in nltk.wordpunct_tokenize(sent) if w.lower() in english_words or not w.isalpha())
print(res)
