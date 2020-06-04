from datetime import datetime
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import textwrap
pd.options.display.width = 0
pd.options.display.max_rows = None
pd.options.display.float_format = "{:.2f}".format


'''The task is to classify the article in two groups politicsNews and worldNews'''


time_now = datetime.now()

'''Reading data from CSV'''
print(time_now, "\n--- Reading data ---\n")
data = pd.read_csv('D:\\Coding_data\\news_true.csv')

print(data.info())
print(data.head())
# print("text to check the beginnings:\n", data['text'][0:50])
print("\nUnique categories of Subject\n", data['subject'].unique())
# print("\nDistribution per category:\n", data.groupby(['subject']).count())
# print("\nFirst title as example:\n", data['title'][0])
# print("\nFirst article as example:\n", "\n".join(textwrap.wrap(data['text'][34], width=100)))

'''CHECK FOR NaNs (generally and in detail)'''
# pd.set_option('use_inf_as_na', True)
# data = data.replace([np.inf, -np.inf], 0).dropna(subset=data.columns, how="all")
#
# missing_values_all = data.isnull().sum().sum()
# print("\nHow many NaNs in dataset?\n", missing_values_all)
#
# missing_values = data.isnull().sum()
# missing_count = (missing_values / data.index.size * 100)
# print("\nWhere are the NaNs:\n", missing_count)

# '''regex'''
# reuters_regex = re.compile(r'((\D+)(\(\D*\)))')
# for i in data['text']:
#     check_reuters = reuters_regex.search(i)
#     print(check_reuters.group(1))

'''Labels transformation'''
cleanup_nums = {"subject": {"politicsNews": 0, "worldnews": 1}}
data.replace(cleanup_nums, inplace=True)
print("\nUnique categories of Subject\n", data['subject'].unique())

'''Throwing out irrelevant features'''
print("--- Throwing out irrelevant features ---\n")
irrelevant_features = ['title', 'date']
data.drop(irrelevant_features, inplace=True, axis=1)

'''Splitting data to train/test'''
print("\n--- Splitting data ---\n")
predict = 'subject'
X = data.drop(predict, axis=1)
y = data[predict].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

'''Applying Bag-of-Words'''
print("\n--- Applying Bag-of-Words ---\n")
vect = CountVectorizer()
X_train = vect.fit_transform(X_train.text)
print("X_train transformed:\n", repr(X_train))

feature_names = vect.get_feature_names()
print("Number of features:\n", len(feature_names))
print("First 20 features:\n", feature_names[:20])
print("Features 20010 to 20030:\n", feature_names[20010:20030])
print("Every 2000th feature:\n", feature_names[::2000])

'''Building a classifier - Logistic Regression'''
print("\n--- Computing Logistic Regression ---\n")
lr = LogisticRegression(max_iter=1000000)
# scores_lr = cross_val_score(lr, X_train, y_train, cv=10)
# print("Mean cross-validation accuracy:\n", np.mean(scores_lr))
'''Mean cross-validation accuracy: 0.9350017939881635'''

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(lr, param_grid, cv=10)
grid.fit(X_train, y_train)
print("Best cross-validation score:\n", grid.best_score_)
print("Best parameters:\n ", grid.best_params_)
'''Best cross-validation score: 0.9427222588597054
C=0,01'''

# lr = LogisticRegression(max_iter=1000000, C=0.01)
X_test = vect.transform(X_test.text)
print("Score on X_test", grid.score(X_test, y_test))
'''Score on X_test 0.9366946778711485'''