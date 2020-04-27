import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

X_digits, y_digits = load_digits(return_X_y=True)

# print("The shape of X:\n", X_digits.shape)
# print("Keys of data:\n", X_digits[0])

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

num_pipeline = Pipeline([
                        ('std_scaler', StandardScaler()),
                    ])

X_train_scaled = num_pipeline.fit_transform(X_train)
X_test_scaled = num_pipeline.fit_transform(X_test)

# log_reg = LogisticRegression(random_state=42, n_jobs=-1)
# log_reg.fit(X_train_scaled, y_train)
# logr_score = log_reg.score(X_test_scaled, y_test)
# print(logr_score)

pipeline = Pipeline([
                    ("kmeans", KMeans(n_clusters=80)),
                    ("log_reg", LogisticRegression(n_jobs=-1, max_iter=5000)),
                    ])
pipeline.fit(X_train_scaled, y_train)
pipl_score = pipeline.score(X_test_scaled, y_test)
print("Pipeline with kmeans + log reg on TEST:\n", pipl_score)

# param_grid = dict(kmeans__n_clusters=range(2, 100))
# grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
# grid_clf.fit(X_train_scaled, y_train)
# print(grid_clf.best_params_)

'''Another use case for clustering is in semi-supervised learning, when we have plenty
of unlabeled instances and very few labeled instances. Let’s train a logistic regression
model on a sample of 50 labeled instances from the digits dataset:'''
# n_labeled = 50
# log_reg_50 = LogisticRegression()
# log_reg_50.fit(X_train_scaled[:n_labeled], y_train[:n_labeled])
# log_reg_50_score = log_reg_50.score(X_test_scaled, y_test)
# print(log_reg_50_score)

'''Let’s see how we can do
better. First, let’s cluster the training set into 50 clusters, then for each cluster let’s find
the image closest to the centroid. We will call these images the representative images:'''
# k = 50
# kmeans = KMeans(n_clusters=k)
# X_digits_dist = kmeans.fit_transform(X_train_scaled)
# representative_digit_idx = np.argmin(X_digits_dist, axis=0)
# X_representative_digits = X_train_scaled[representative_digit_idx]
#
# plt.figure(figsize=(8, 2))
# for index, X_representative_digit in enumerate(X_representative_digits):
#     plt.subplot(k // 10, 10, index + 1)
#     plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
#     plt.axis('off')
# plt.show()
#
# y_representative_digits = np.array([
#     1, 1, 9, 5, 4, 9, 6, 0, 3, 3, 2, 7, 4, 2, 4, 5, 4, 3, 2, 6,
#     3, 7, 4, 8, 5, 5, 4, 1, 6, 7, 2, 4, 8, 9, 8, 4, 5, 8, 9, 3,
#     7, 9, 2, 7, 4, 5, 6, 4, 7, 8
# ])
#
# log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42, n_jobs=-1)
# log_reg.fit(X_representative_digits, y_representative_digits)
# log_reg_2 = log_reg.score(X_test_scaled, y_test)
# print(log_reg_2)

'''Lets try DBSCAN'''
dbscan = DBSCAN(eps=5, n_jobs=-1)
dbscan.fit(X_train_scaled)
# print(dbscan.labels_[:100])
print("DBSCAN on TRAIN how many indices:\n", len( dbscan.core_sample_indices_))


knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

knn_score = knn.score(X_train_scaled, y_train)
print("Score for dbscan + knn on TRAIN:\n", knn_score)

'''Lets try PCA principal component analysis'''
knn_2 = KNeighborsClassifier()
knn_2.fit(X_train_scaled, y_train)

knn_score_2 = knn_2.score(X_test_scaled, y_test)
print("Score for knn on TEST:\n", knn_score_2)

'''Mix PCA with knn'''
pca = PCA(n_components=20, whiten=True, random_state=0)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn_pca = KNeighborsClassifier(n_neighbors=1)
knn_pca.fit(X_train_pca, y_train)

knn_score_pca = knn_pca.score(X_test_pca, y_test)
print("Score for pca + knn on TEST:\n", knn_score_pca)




