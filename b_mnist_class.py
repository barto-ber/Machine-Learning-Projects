import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
pd.options.display.width = 0
pd.options.display.max_rows = None

# train_data = pd.read_csv('mnist_train.csv')
# test_data = pd.read_csv('mnist_test.csv')
#
# print(train_data.head())
# # print(train_data.info())
#
# X_train = train_data.drop("5", axis=1)
# y_train = train_data["5"].copy()
# X_test = test_data.drop("5", axis=1)
# y_test = test_data["5"].copy()
#
#
# num_pipeline = Pipeline([
#                         ('std_scaler', StandardScaler()),
#                     ])
#
# X_train_scaled = num_pipeline.fit_transform(X_train)
# X_test_scaled = num_pipeline.fit_transform(X_test)

# log_reg = LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000)
# log_reg.fit(X_train_scaled, y_train)
# logreg_score = log_reg.score(X_test_scaled, y_test)
# print("Score for Logistic Regression:\n", logreg_score)

# pipeline = Pipeline([
#                     ("kmeans", KMeans(n_clusters=50, n_jobs=-1)),
#                     ("log_reg", LogisticRegression(n_jobs=-1, max_iter=1000))
#                     ])
# pipeline.fit(X_train_scaled, y_train)
# pipl_score = pipeline.score(X_test_scaled, y_test)
# print("Score for Pipeline:\n", pipl_score)



image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
train_data = np.loadtxt("mnist_train.csv", delimiter=",")
test_data = np.loadtxt("mnist_test.csv", delimiter=",")
# print(train_data[:10])

'''The images of the MNIST dataset are greyscale and the pixels range between 0 and 255 including both bounding values.
We will map these values into an interval from [0.01, 1] by multiplying each pixel by 0.99 / 255 and adding 0.01 to the result.
This way, we avoid 0 values as inputs, which are capable of preventing weight updates, as we we seen in the introductory chapter'''
fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

'''We need the labels in our calculations in a one-hot representation. We have 10 digits from 0 to 9, i.e. lr = np.arange(10).
Turning a label into one-hot representation can be achieved with the command: (lr==label).astype(np.int)'''
lr = np.arange(10)

for label in range(10):
    one_hot = (lr==label).astype(np.int)
    # print("label: ", label, " in one-hot representation: ", one_hot)

'''We are ready now to turn our labelled images into one-hot representations. Instead of zeroes and one,
we create 0.01 and 0.99, which will be better for our calculations:'''
lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

# for i in range(3):
#     print(train_imgs)
#     img = train_imgs[i].reshape((28,28))
#     plt.imshow(img, cmap="Greys")
#     plt.show()

# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(train_imgs, train_labels)
# logreg_score = log_reg.score(test_imgs, test_labels)
# print("Score for Logistic Regression:\n", logreg_score)

# pipeline = Pipeline([
#                     ("kmeans", KMeans(n_clusters=50, n_jobs=-1)),
#                     ("log_reg", LogisticRegression(n_jobs=-1, max_iter=1000))
#                     ])
# pipeline.fit(train_imgs, train_labels_one_hot)
# pipl_score = pipeline.score(test_imgs, test_labels_one_hot)
# print("Score for Pipeline:\n", pipl_score)

'''Lets check a binary classifier SGD'''
some_digit = train_imgs[0] # some digit is 5

# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, interpolation="nearest")
# plt.axis("off")
# plt.show()

# print("Some digit:\n", some_digit)
some_label = train_labels[0]
# print("Some label:\n", some_label)
y_train_5 = (train_labels == 5)
# print(y_train_4)
y_test_5 = (test_labels == 5)

sgd_5 = SGDClassifier(random_state=42)
sgd_5.fit(train_imgs, y_train_5)
sgd_5_predict = sgd_5.predict(([some_digit]))
print("\nIs some_digit 5?:\n", sgd_5_predict)

acc_sgd = cross_val_score(sgd_5, train_imgs, y_train_5, cv=3, scoring="accuracy")
print("Cross Val Score for SGD is:\n", acc_sgd)

y_train_pred = cross_val_predict(sgd_5, train_imgs, y_train_5, cv=3)

conf_matr_sgd = confusion_matrix(y_train_5, y_train_pred)
print("Confusion matrix for sgd:\n", conf_matr_sgd)

prec_score_sgd = precision_score(y_train_5, y_train_pred)
print("Precision score for sgd (how often is the result correct):\n", prec_score_sgd)
rec_score_sgd = recall_score(y_train_5, y_train_pred)
print("Recall score for sgd (how many some_digits it detects):\n", rec_score_sgd)

f1_sc_sgd = f1_score(y_train_5, y_train_pred)
print("F1 score for sgd (harmonic mean of precision and recall:\n", f1_sc_sgd)

y_5_scores = cross_val_predict(sgd_5, train_imgs, y_train_5, cv=3, method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_5_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

