import pandas as pd
import numpy as np

train_dataset = pd.read_csv("test.csv")

print(train_dataset.columns[train_dataset.isna().any()].tolist())

lis = []
for i in train_dataset.columns:
    lis.append(i)
for j in range(len(lis)):
    print(j, lis[j])