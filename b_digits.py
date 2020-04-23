from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_digits, y_digits = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

# log_reg = LogisticRegression(random_state=42, n_jobs=-1, max_iter=50000)
# log_reg.fit(X_train, y_train)
# logr_score = log_reg.score(X_test, y_test)
# print(logr_score)

pipeline = Pipeline([
("kmeans", KMeans(n_clusters=70)),
("log_reg", LogisticRegression(n_jobs=-1, max_iter=50000)),
])
pipeline.fit(X_train, y_train)
pipl_score = pipeline.score(X_test, y_test)
print(pipl_score)

# param_grid = dict(kmeans__n_clusters=range(2, 100))
# grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
# grid_clf.fit(X_train, y_train)
# print(grid_clf.best_params_)




