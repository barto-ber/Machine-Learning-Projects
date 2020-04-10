import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
pd.options.display.width = 0
pd.options.display.max_rows = None


data = pd.read_csv('car.data')

# print(data.info())
# print(data.head())
# print("\nHow many NaN in dataset?\n", data.isnull().sum().sum())


train_set, test_set = train_test_split(data, test_size=0.15, random_state=42)

predict = "class"

X_train = train_set.drop(predict, axis=1)
y_train = train_set[predict].copy()


# FULL PIPELINE
cat_attribs = ["buying", "maint", "door", "persons", "lug_boot", "safety"]
full_pipeline = ColumnTransformer([
                                    ("cat", OneHotEncoder(), cat_attribs),
                                ])

X_train_prepared = full_pipeline.fit_transform(X_train)


model = KNeighborsClassifier(n_neighbors=9)

model.fit(X_train_prepared, y_train)

acc_train = model.score(X_train_prepared, y_train)
print("\nScore for K Neighbors Classifier training set:\n", acc_train)

X_test = test_set.drop(predict, axis=1)
y_test = test_set[predict].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = model.predict(X_test_prepared)

acc_test = model.score(X_test_prepared, y_test)
print("\nScore for K Neighbors Classifier test set:\n", acc_test)


# names = np.array(["acc", "good", "unacc", "vgood"], dtype=str)
#
# for x in range(len(final_predictions)):
#     print("Predicted: ", names[final_predictions[x]], "Data: ", X_test_prepared[x], "Actual: ", names[y_test[x]])
#     n = model.kneighbors([X_train_prepared[x]], 9, True)
#     print("N: ", n)