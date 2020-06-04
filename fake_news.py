from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
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
print("text to check the beginnings:\n", data['text'][0:50])
print("\nUnique categories of Subject\n", data['subject'].unique())
# print("\nDistribution per category:\n", data.groupby(['subject']).count())
# print("\nFirst title as example:\n", data['title'][0])
print("\nFirst article as example:\n", "\n".join(textwrap.wrap(data['text'][34], width=100)))

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

# '''Throwing out irrelevant features'''
# print("--- Throwing out irrelevant features ---\n")
# irrelevant_features = ['title', 'date']
# data.drop(irrelevant_features, inplace=True, axis=1)
#
# '''Splitting data to train/test'''
# print("\n--- Splitting data ---\n")
# predict = 'subject'
# X = data.drop(predict, axis=1)
# y = data[predict].copy()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#
# '''Applying Bag-of-Words'''
# vect = CountVectorizer()
# vect.fit(X_train)

