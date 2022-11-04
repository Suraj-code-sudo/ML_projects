import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', index_col=False)
test = pd.read_csv('test.csv', index_col=False)

train = train.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]

for df in combine:
    df['FamilySize'] = df['SibSp']+df['Parch']
print(train['FamilySize'])