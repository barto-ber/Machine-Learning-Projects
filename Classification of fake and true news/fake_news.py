from datetime import datetime
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
    precision_recall_curve, roc_curve, precision_score, recall_score, \
    classification_report, roc_auc_score
import textwrap
import matplotlib.pyplot as plt
import mglearn
pd.options.display.width = 0
pd.options.display.max_rows = None
pd.options.display.float_format = "{:.2f}".format


'''The task is to classify the article in two groups politicsNews and worldNews.

PART 1 - CountVectorizer, TF-IDF & Log Reg
Conclusion: First try was to apply Bag of words with default settings as well as Logistic Regression with default settings.
The cross validation accuracy was 0.935, not bad. Using GridSearchCV and finding best params was possible to push 
the cross validation accuracy to 0.942. Testing this best model on the test set got accuracy of 0.936, not bad.

Trying to get more using Stop-words in Countvectorizer got the cross validation accuracy of 0.942 same as without Stop words.
Last try for rescaling features with TF-IDF with different n_grams-values got the cross validation of 0.946, 
kind of the same as other methods.


PART 2 - Latent Dirichlet Allocation (LDA)
Conclusion: This approach was quite successful; it seems that the algorithm recognized two groups politics news amd world news.
We can see it in the print of some topic words.

PRINT OF ARTICLES NOT POSSIBLE?????
'''

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

'''regex to clean up the first words of each article which are the name of the source'''
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



'''
---PART 1
Applying Bag-of-Words'''
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
'''===Mean cross-validation accuracy: 0.9350017939881635'''

# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
# grid = GridSearchCV(lr, param_grid, cv=10)
# grid.fit(X_train, y_train)
# print("Best cross-validation score:\n", grid.best_score_)
# print("Best parameters:\n ", grid.best_params_)
'''===Best cross-validation score: 0.9427222588597054
C=0,01'''

# lr = LogisticRegression(max_iter=1000000, C=0.01)
# X_test = vect.transform(X_test.text)
# print("Score on X_test", grid.score(X_test, y_test))
'''===Score on X_test 0.9366946778711485'''


'''Applying Bag-of-Words with STOPWORDS'''
# print("\n--- Applying Bag-of-Words with STOPWORDS ---\n")
# vect = CountVectorizer(min_df=5, stop_words="english")
# X_train = vect.fit_transform(X_train.text)
# print("X_train with stop words:\n", repr(X_train))
#
# grid = GridSearchCV(lr, param_grid, cv=10)
# grid.fit(X_train, y_train)
# print("Best cross-validation score with Stopwords:\n", grid.best_score_)
'''===Best cross-validation score with Stopwords: 0.9429712473681068'''


'''Model with rescaling with tf-idf'''
# print("\n--- Computing Logistic Regression with tf-idf ---\n")
# pipe = make_pipeline(TfidfVectorizer(min_df=0.00025, norm=None), lr)
# param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
# grid = GridSearchCV(pipe, param_grid, cv=10)
# grid.fit(X_train.text, y_train)
# print("Best parameters tf-idf:\n ", grid.best_params_)
# print("Best cross-validation score with tf-idf:\n", grid.best_score_)
'''
===min_df=0.5
Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.7029631802334277
 
===min_df=0.2
Best parameters tf-idf:
  {'logisticregression__C': 0.01}
Best cross-validation score with tf-idf:
 0.8973980584630908

===min_df=0.1
Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.923982405742002

===min_df=0.05
Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.9325742528988602
 
===min_df=0.001
 Best parameters tf-idf:
  {'logisticregression__C': 0.001}
Best cross-validation score with tf-idf:
 0.9458353514085713
 
===min_df=0.00025
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
===Best parameters with n-grams:
 {'logisticregression__C': 10, 'tfidfvectorizer__ngram_range': (1, 1)}
===Best cross-validation score with n-grams:
 0.9468310729599099
'''


'''First best model CHECK'''
# print("\n--- Computing First best model ---\n")
# lr_best = LogisticRegression(max_iter=1000000, C=0.01)
# lr_best.fit(X_train, y_train)
#
# predictions_lr_best = cross_val_predict(lr_best, X_train, y_train, cv=10)
# c_matr = confusion_matrix(y_train, predictions_lr_best)
# print("Confusion matrix:\n", c_matr)
'''
===Confusion matrix:
 [[7930  525]
 [ 395 7212]]
'''


'''Looking for better model Search grid CountVectorizer plus Log reg'''
# print("\n--- Computing Search grid CountVectorizer plus Log reg ---\n")
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
===Best parameters with search grid Vect and Lr:
 {'lr__C': 0.01, 'vect__max_df': 0.75}
===Best cross-validation score with search grid Vect and Lr:
 0.9430958578634414
'''
'''Using best params for Countvect for best model and CHECK'''
# print("\n--- Best params - Applying Bag-of-Words ---\n")
# vect = CountVectorizer(max_df=0.75)
# X_train = vect.fit_transform(X_train.text)
# X_test = vect.transform((X_test.text))
#
# print("\n--- Computing Best Model ---\n")

# lr_best.fit(X_train, y_train)

# predictions_lr_best = cross_val_predict(lr_best, X_train, y_train,
#                                         cv=5, n_jobs=-1, method='decision_function')
# c_matr = confusion_matrix(y_train, predictions_lr_best)
# print("Confusion matrix:\n", c_matr)
'''
Confusion matrix:
 [[7936  519]
 [ 395 7212]]
'''
# plot_confusion_matrix(lr_best, X_train, y_train)
# plt.show()


'''TESTING best model ON THE TEST DATA'''
# print("\n--- Testing Best Model ---\n")
# predictions_lr_best_test = lr_best.predict(X_test)
# c_matr = confusion_matrix(y_test, predictions_lr_best_test)
# print("Test best model confusion matrix:\n", c_matr)
# plot_confusion_matrix(lr_best, X_test, y_test, values_format='.2f')
'''
Test best model confusion matrix:
 [[2638  179]
 [ 152 2386]]
'''

# classsif_rep_lr_best_test = classification_report(y_test, predictions_lr_best_test)
# print("Classification report test best model:\n", classsif_rep_lr_best_test)
'''
               precision    recall  f1-score   support

           0       0.95      0.94      0.94      2817
           1       0.93      0.94      0.94      2538

    accuracy                           0.94      5355
   macro avg       0.94      0.94      0.94      5355
weighted avg       0.94      0.94      0.94      5355
'''

# auc = roc_auc_score(y_test, lr_best.decision_function(X_test))
# print("AUC of test best model:\n", auc)
'''
AUC of test best model:
 0.9803684737464449
'''

# precisions, recalls, thresholds = precision_recall_curve(y_test, predictions_lr_best_test)
# def plot_precision_vs_recall(precisions, recalls):
#     plt.plot(recalls, precisions, "b-", linewidth=2)
#     plt.xlabel("Recall", fontsize=16)
#     plt.ylabel("Precision", fontsize=16)
#     plt.axis([0, 1, 0, 1])
#     plt.grid(True)
#
# plt.figure(figsize=(8, 6))
# plot_precision_vs_recall(precisions, recalls)
#
# fpr, tpr, threshs = roc_curve(y_test, predictions_lr_best_test)
# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
#     plt.ylabel('True Positive Rate (Recall)', fontsize=16)
#     plt.grid(True)
#
# plt.figure(figsize=(8, 6))
# plot_roc_curve(fpr, tpr)
#
# plt.show()



'''
---PART 2
Latent Dirichlet Allocation (LDA)'''
print("--- Computing Latent Dirichlet Allocation (LDA) ---")
vect_lda = CountVectorizer(max_features=10000, max_df=0.25)
X_lda = vect_lda.fit_transform(X_train.text)

lda = LatentDirichletAllocation(n_components=2, learning_method="batch",
                                max_iter=25, random_state=42, n_jobs=-1)
# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once
document_topics = lda.fit_transform(X_lda)

print("LDA shape of components:\n", lda.components_.shape)

# For each topic (a row in the components_), sort the features (ascending)
# Invert rows with [:, ::-1] to make sorting descending
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names from the vectorizer
feature_names = np.array(vect_lda.get_feature_names())
# Print out the 10 topics:
mglearn.tools.print_topics(topics=range(2), feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=5, n_words=20)
'''
topic 0       topic 1       
--------      --------      
north         campaign      
minister      clinton       
china         senate        
military      obama         
korea         percent       
security      white         
party         tax           
foreign       democratic    
iran          court         
police        presidential  
nuclear       she           
eu            bill          
russia        law           
officials     republicans   
war           administration
countries     her           
south         federal       
european      party         
prime         committee     
she           congress 
'''

# sort by weight of "world news" topic 0
music = np.argsort(document_topics[:, 0])[::-1]
# print the five documents where the topic is most important
for i in music[:10]:
    # pshow first two sentences
    print("\n".join(X_train.text[i]))#can not PRINT OUT!!!!!!





