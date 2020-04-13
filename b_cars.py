import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
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

X_test = test_set.drop(predict, axis=1)
y_test = test_set[predict].copy()

X_test_prepared = full_pipeline.transform(X_test)


best_prediction = 0
k_range = range(1, 26)
scores_train = []
scores_test = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train_prepared, y_train)

    acc_train = model.score(X_train_prepared, y_train)
    scores_train.append(acc_train)
    # print("\nScore for K Neighbors Classifier training set:\n", k, acc_train)

    final_predictions = model.predict(X_test_prepared)

    acc_test = model.score(X_test_prepared, y_test)
    scores_test.append(acc_test)
    # print("\nScore for K Neighbors Classifier test set:\n", k, acc_test)

    # If the current model has a better score than one we've already trained then save it
    if acc_test > best_prediction:
        best_prediction = acc_test
        with open("cars_class.pickle", "wb") as f:
            pickle.dump(model, f)

print("My for-loop scores:\n", *scores_test, sep="\n")
print("\nBest prediction from for loop:\n", best_prediction)

# LOAD MODEL
pickle_in = open("cars_class.pickle", "rb")
model_loaded = pickle.load(pickle_in)
acc_test_loaded = model_loaded.score(X_test_prepared, y_test)
print("\nLoaded model score:\n", acc_test_loaded)

cvs_test = cross_val_score(model_loaded, X_test_prepared, y_test, cv=5, scoring="accuracy")
print("\nCross validation score on the test set:\n", *cvs_test, sep="\n")

cvp_test = cross_val_predict(model_loaded, X_test_prepared, y_test, cv=5)
print("\nCross validation predict on the test set:\n", *cvs_test, sep="\n")

# PLOT ALL MODELS FROM FOR LOOP
# fig, ax = plt.subplots()
# ax.plot(k_range, scores_train, label="Scores train set")
# ax.plot(k_range, scores_test, label="Scores test set")
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Testing Accuracy')
# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# plt.legend()
# plt.show()

