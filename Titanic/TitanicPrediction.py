import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score

### Getting Files
train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")

### Handling Missing values
test_dataset.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)
train_dataset.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)

train_dataset['Age'].fillna(train_dataset['Age'].median(), inplace=True)
test_dataset['Age'].fillna(test_dataset['Age'].median(), inplace=True)
train_dataset['Embarked'].fillna('S', inplace=True)
test_dataset["Fare"].fillna(test_dataset["Fare"].median(), inplace = True)

### Splitting X and y
features = ["Pclass", "Sex", "Age", "Embarked", "Fare"]
X_train = train_dataset[features]
y_train = train_dataset["Survived"]
X_test = test_dataset[features]

X_train.loc[X_train['Sex'] == 'male', 'Sex'] = 0
X_train.loc[X_train['Sex'] == 'female', 'Sex'] = 1
X_train.loc[X_train['Embarked'] == 'S', 'Embarked'] = 0
X_train.loc[X_train['Embarked'] == 'C', 'Embarked'] = 1
X_train.loc[X_train['Embarked'] == 'Q', 'Embarked'] = 2

X_test.loc[X_test['Sex'] == 'male', 'Sex'] = 0
X_test.loc[X_test['Sex'] == 'female', 'Sex'] = 1
X_test.loc[X_test['Embarked'] == 'S', 'Embarked'] = 0
X_test.loc[X_test['Embarked'] == 'C', 'Embarked'] = 1
X_test.loc[X_test['Embarked'] == 'Q', 'Embarked'] = 2

from sklearn.model_selection import train_test_split
X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
regressor = AdaBoostClassifier(n_estimators=100, random_state=0)
regressor.fit(X_training, y_training)

y_pred = regressor.predict(X_test)

y_pred = y_pred.round().astype(int)
df = pd.DataFrame(y_pred, columns=["Survived"])
df.to_csv("E:/KaggleDataSets/Titanic/predicted.csv")
