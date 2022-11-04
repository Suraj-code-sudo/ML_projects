import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("E:/KaggleDataSets/Disease Prediction/breast-cancer.csv")
dia_y = df['diagnosis']
print(df['diagnosis'].value_counts()['M'])
print(df['diagnosis'].value_counts()['B'])
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
radius_mean = df.iloc[:, 2].values

columns_values = (df.columns)
cols = df.columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
classifier = HistGradientBoostingClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(y_test)
print(y_pred)

corr=df.corr()
f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(corr,annot=True,annot_kws = {'size':4} ,linewidths=1,fmt=".1f",ax=ax,cmap="YlGnBu",square=True)
#plt.show()

for col in df.columns:
    plt.scatter(col, y,color='blue')
    plt.title('Breast Cancer Predictions')
    plt.xlabel('Diagnosis')
    plt.ylabel(col)
    #plt.show()