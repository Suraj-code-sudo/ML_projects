import pandas as pd
import numpy as np

train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")

del train_dataset['PoolQC']
del test_dataset['PoolQC']

X = train_dataset.iloc[:, :-1].values
y = train_dataset.iloc[:, -1].values
test = test_dataset.iloc[:, :].values


### Handling Missing values
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
most_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
const_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Empty')

X[:, 3:4] = mean_imputer.fit_transform(X[:, 3:4])
X[:, 26:27] = mean_imputer.fit_transform(X[:, 26:27])
X[:, 59:60] = mean_imputer.fit_transform(X[:, 59:60])

X[:, 6:7] = const_imputer.fit_transform(X[:, 6:7])
X[:, 25:26] = const_imputer.fit_transform(X[:, 25:26])
X[:, 30:34] = const_imputer.fit_transform(X[:, 30:34])
X[:, 35:36] = const_imputer.fit_transform(X[:, 35:36])
X[:, 57:59] = const_imputer.fit_transform(X[:, 57:59])
X[:, 60:61] = const_imputer.fit_transform(X[:, 60:61])

X[:, 42:43] = most_imputer.fit_transform(X[:, 42:43])
X[:, 63:65] = most_imputer.fit_transform(X[:, 63:65])
X[:, 72:74] = const_imputer.fit_transform(X[:, 72:74])

### Test Data

test[:, 3:4] = mean_imputer.fit_transform(test[:, 3:4])
test[:, 26:27] = mean_imputer.fit_transform(test[:, 26:27])
test[:, 34:35] = mean_imputer.fit_transform(test[:, 34:35])
test[:, 36:37] = mean_imputer.fit_transform(test[:, 36:37])
test[:, 37:39] = mean_imputer.fit_transform(test[:, 37:39])
test[:, 47:49] = mean_imputer.fit_transform(test[:, 47:49])
test[:, 59:60] = mean_imputer.fit_transform(test[:, 59:60])
test[:, 61:63] = mean_imputer.fit_transform(test[:, 61:63])

test[:, 6:7] = const_imputer.fit_transform(test[:, 6:7])
test[:, 23:26] = const_imputer.fit_transform(test[:, 23:26])
test[:, 30:34] = const_imputer.fit_transform(test[:, 30:34])
test[:, 53:54] = const_imputer.fit_transform(test[:, 53:54])
test[:, 57:59] = const_imputer.fit_transform(test[:, 57:59])
test[:, 60:61] = const_imputer.fit_transform(test[:, 60:61])
test[:, 63:65] = const_imputer.fit_transform(test[:, 63:65])
test[:, 72:74] = const_imputer.fit_transform(test[:, 72:74])
test[:, 77:78] = const_imputer.fit_transform(test[:, 77:78])

test[:, 2:3] = most_imputer.fit_transform(test[:, 2:3])
test[:, 9:10] = most_imputer.fit_transform(test[:, 9:10])
test[:, 35:36] = most_imputer.fit_transform(test[:, 35:36])
test[:, 42:43] = most_imputer.fit_transform(test[:, 42:43])
test[:, 55:56] = most_imputer.fit_transform(test[:, 55:56])




### Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, 3:5] = sc.fit_transform(X[:, 3:5])
X[:, 19:21] = sc.fit_transform(X[:, 19:21])
X[:, 26:27] = sc.fit_transform(X[:, 26:27])
X[:, 34:35] = sc.fit_transform(X[:, 34:35])
X[:, 37:39] = sc.fit_transform(X[:, 37:39])
X[:, 43:45] = sc.fit_transform(X[:, 43:45])
X[:, 46:47] = sc.fit_transform(X[:, 46:47])
X[:, 59:60] = sc.fit_transform(X[:, 59:60])
X[:, 62:63] = sc.fit_transform(X[:, 62:63])
X[:, 66:71] = sc.fit_transform(X[:, 66:71])
X[:, 74:77] = sc.fit_transform(X[:, 74:77])

test[:, 3:5] = sc.fit_transform(test[:, 3:5])
test[:, 19:21] = sc.fit_transform(test[:, 19:21])
test[:, 26:27] = sc.fit_transform(test[:, 26:27])
test[:, 34:35] = sc.fit_transform(test[:, 34:35])
test[:, 37:39] = sc.fit_transform(test[:, 37:39])
test[:, 43:45] = sc.fit_transform(test[:, 43:45])
test[:, 46:47] = sc.fit_transform(test[:, 46:47])
test[:, 59:60] = sc.fit_transform(test[:, 59:60])
test[:, 62:63] = sc.fit_transform(test[:, 62:63])
test[:, 66:71] = sc.fit_transform(test[:, 66:71])
test[:, 74:77] = sc.fit_transform(test[:, 74:77])


### Label Encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
oc = OrdinalEncoder()
encoder = LabelEncoder()

X[:, 2] = encoder.fit_transform(X[:, 2])
X[:, 35] = encoder.fit_transform(X[:, 35])

X[:, 5:17] = oc.fit_transform(X[:, 5:17])
X[:, 21:26] = oc.fit_transform(X[:, 21:26])
X[:, 27:34] = oc.fit_transform(X[:, 27:34])
X[:, 39:43] = oc.fit_transform(X[:, 39:43])
X[:, 63:66] = oc.fit_transform(X[:, 63:66])
X[:, 72:74] = oc.fit_transform(X[:, 72:74])
X[:, 77:79] = oc.fit_transform(X[:, 77:79])

X[:, 53] = encoder.fit_transform(X[:, 53])
X[:, 55] = encoder.fit_transform(X[:, 55])
X[:, 57] = encoder.fit_transform(X[:, 57])
X[:, 58] = encoder.fit_transform(X[:, 58])
X[:, 60] = encoder.fit_transform(X[:, 60])
X[:, 61] = encoder.fit_transform(X[:, 61])
X[:, 35] = encoder.fit_transform(X[:, 35])


test[:, 2] = encoder.fit_transform(test[:, 2])
test[:, 35] = encoder.fit_transform(test[:, 35])

test[:, 5:17] = oc.fit_transform(test[:, 5:17])
test[:, 21:26] = oc.fit_transform(test[:, 21:26])
test[:, 27:34] = oc.fit_transform(test[:, 27:34])
test[:, 39:43] = oc.fit_transform(test[:, 39:43])
test[:, 63:66] = oc.fit_transform(test[:, 63:66])
test[:, 72:74] = oc.fit_transform(test[:, 72:74])
test[:, 77:79] = oc.fit_transform(test[:, 77:79])

test[:, 53] = encoder.fit_transform(test[:, 53])
test[:, 55] = encoder.fit_transform(test[:, 55])
test[:, 57] = encoder.fit_transform(test[:, 57])
test[:, 58] = encoder.fit_transform(test[:, 58])
test[:, 60] = encoder.fit_transform(test[:, 60])
test[:, 61] = encoder.fit_transform(test[:, 61])
test[:, 35] = encoder.fit_transform(test[:, 35])

test_r = pd.DataFrame(X)
test_r.to_csv("E:/KaggleDataSets/House Pricing Advanced Regression Techniques/results.csv")
### Modelling

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(test)

result = pd.DataFrame(y_pred, columns=['SalePrice'])
#print(result)
result.to_csv("E:/KaggleDataSets/House Pricing Advanced Regression Techniques/results.csv")