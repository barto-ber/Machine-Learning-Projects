from datetime import datetime
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import textwrap
import matplotlib.pyplot as plt
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
# print("\n--- Applying Bag-of-Words ---\n")
# vect = CountVectorizer()
# X_train = vect.fit_transform(X_train.text)
# print("X_train transformed:\n", repr(X_train))
#
# feature_names = vect.get_feature_names()
# print("Number of features:\n", len(feature_names))
# print("First 20 features:\n", feature_names[:20])
# print("Features 20010 to 20030:\n", feature_names[20010:20030])
# print("Every 2000th feature:\n", feature_names[::2000])

'''Building a classifier - Logistic Regression'''
# print("\n--- Computing Logistic Regression ---\n")
# lr = LogisticRegression(max_iter=1000000)
# scores_lr = cross_val_score(lr, X_train, y_train, cv=10)
# print("Mean cross-validation accuracy:\n", np.mean(scores_lr))
'''Mean cross-validation accuracy: 0.9350017939881635'''

# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
# grid = GridSearchCV(lr, param_grid, cv=10)
# grid.fit(X_train, y_train)
# print("Best cross-validation score:\n", grid.best_score_)
# print("Best parameters:\n ", grid.best_params_)
'''Best cross-validation score: 0.9427222588597054
C=0,01'''

# lr = LogisticRegression(max_iter=1000000, C=0.01)
# X_test = vect.transform(X_test.text)
# print("Score on X_test", grid.score(X_test, y_test))
'''Score on X_test 0.9366946778711485'''


'''Applying Bag-of-Words with STOPWORDS'''
# print("\n--- Applying Bag-of-Words with STOPWORDS ---\n")
# vect = CountVectorizer(min_df=5, stop_words="english")
# X_train = vect.fit_transform(X_train.text)
# print("X_train with stop words:\n", repr(X_train))
#
# grid = GridSearchCV(lr, param_grid, cv=10)
# grid.fit(X_train, y_train)
# print("Best cross-validation score with Stopwords:\n", grid.best_score_)
'''Best cross-validation score with Stopwords: 0.9429712473681068'''


'''Model with rescaling with tf-idf'''
# print("\n--- Computing Logistic Regression with tf-idf ---\n")
# pipe = make_pipeline(TfidfVectorizer(min_df=0.00025, norm=None), lr)
# param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
# grid = GridSearchCV(pipe, param_grid, cv=10)
# grid.fit(X_train.text, y_train)
# print("Best parameters tf-idf:\n ", grid.best_params_)
# print("Best cross-validation score with tf-idf:\n", grid.best_score_)
'''
-min_df=0.5
Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.7029631802334277
 
-min_df=0.2
Best parameters tf-idf:
  {'logisticregression__C': 0.01}
Best cross-validation score with tf-idf:
 0.8973980584630908

-min_df=0.1
Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.923982405742002

-min_df=0.05
Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.9325742528988602
 
 -min_df=0.001
 Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.9458353514085713
 
 min_df=0.00025
 Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.9462088729182181
'''


'''Bag-of-Words with More Than One Word (n-Grams)'''
# print("\n--- Computing Logistic Regression with n-Grams ---\n")
# pipe = make_pipeline(TfidfVectorizer(min_df=0.001), lr)
# # running the grid search takes a long time because of the
# # relatively large grid and the inclusion of trigrams
# param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
#               "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
# grid = GridSearchCV(pipe, param_grid, cv=10)
# grid.fit(X_train.text, y_train)
# print("Best parameters with n-grams:\n", grid.best_params_)
# print("Best cross-validation score with n-grams:\n", grid.best_score_)
'''
Best parameters with n-grams:
 {'logisticregression__C': 10, 'tfidfvectorizer__ngram_range': (1, 1)}
Best cross-validation score with n-grams:
 0.9468310729599099
'''

'''Best model CHECK'''
# print("\n--- Computing Best Model ---\n")
# lr_best = LogisticRegression(max_iter=1000000, C=0.01)
# lr_best.fit(X_train, y_train)
#
# predictions_lr_best = cross_val_predict(lr_best, X_train, y_train, cv=10)
# c_matr = confusion_matrix(y_train, predictions_lr_best)
# print("Confusion matrix:\n", c_matr)
'''
Confusion matrix:
 [[7930  525]
 [ 395 7212]]
'''


'''Search grid COuntVectorizer plus Log reg'''
# print("\n--- Computing Search grid COuntVectorizer plus Log reg ---\n")
# pipe = Pipeline([('vect', CountVectorizer()),
#                  ('lr', LogisticRegression(max_iter=1000000))
#                  ])
# param_grid = [{'vect__max_df': [0.5, 0.75, 1.0],
#               'lr__C': [0.001, 0.01, 0.1, 1, 10, 100]
#               }]
# grid = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
#
# grid.fit(X_train.text, y_train)
# print("Best parameters with search grid Vect and Lr:\n", grid.best_params_)
# print("Best cross-validation score with search grid Vect and Lr:\n", grid.best_score_)
'''
Best parameters with search grid Vect and Lr:
 {'lr__C': 0.01, 'vect__max_df': 0.75}
Best cross-validation score with search grid Vect and Lr:
 0.9430958578634414
'''
print("\n--- Applying Bag-of-Words ---\n")
vect = CountVectorizer(max_df=0.75)
X_train = vect.fit_transform(X_train.text)
print("\n--- Computing Best Model ---\n")
lr_best = LogisticRegression(max_iter=1000000, C=0.01, n_jobs=-1, random_state=42)
lr_best.fit(X_train, y_train)

predictions_lr_best = cross_val_predict(lr_best, X_train, y_train, cv=10, n_jobs=-1)
c_matr = confusion_matrix(y_train, predictions_lr_best)
print("Confusion matrix:\n", c_matr)

'''
Confusion matrix:
 [[7936  519]
 [ 395 7212]]
'''

plot_confusion_matrix(lr_best, X_train, y_train)  # doctest: +SKIP
plt.show()  # doctest: +SKIP
