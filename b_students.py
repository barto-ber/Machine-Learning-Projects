import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
pd.options.display.width = 0
pd.options.display.max_rows = None

dtypes = {"sex": "category",
          "address": "category",
          "famsize": "category",
          "Pstatus": "category",
          "schoolsup": "category",
          "paid": "category",
          "higher": "category"}

data = pd.read_csv('student-mat.csv', dtype=dtypes, sep=";")
data = data[["sex", "address", "famsize", "Pstatus", "Mother edu", "Father edu", "studytime", "failures", "schoolsup",
             "paid", "higher", "absences", "G1", "G2", "G3"]]
# print(data.info())
# print("\nHow many NaN in dataset?\n", data.isnull().sum().sum())

train_set, test_set = train_test_split(data, test_size=0.15, random_state=42)

corr_matrix = train_set.corr()
show_correlations = corr_matrix["G3"].sort_values(ascending=False)
# print("\nCorrelations between every pair of attributes:\n", show_correlations)

# attributes = ["studytime", "Father edu", "G3"]
# scatter_matrix(train_set[attributes])
# plt.show()

X_train = train_set.drop("G3", axis=1)
y_train = train_set["G3"].copy()

# predict = "G3"

# X = np.array(data.drop([predict], axis=1))
# y = np.array(data[predict])
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_train_num = X_train.drop(["sex", "address", "famsize", "Pstatus", "schoolsup",
                            "paid", "higher"], axis=1)
num_pipeline = Pipeline([
                        # ('imputer', SimpleImputer(strategy="median")),
                        ('std_scaler', StandardScaler()),
                    ])

X_train_num_scaled = num_pipeline.fit_transform(X_train_num)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs = list(X_train_num)
cat_attribs = ["sex", "address", "famsize", "Pstatus", "schoolsup", "paid", "higher"]
full_pipeline = ColumnTransformer([
                                    ("num", num_pipeline, num_attribs),
                                    ("cat", OneHotEncoder(), cat_attribs),
                                ])

X_train_prepared = full_pipeline.fit_transform(X_train)


lin_regr = linear_model.LinearRegression()
lin_regr.fit(X_train_prepared, y_train)

acc_train = lin_regr.score(X_train_prepared, y_train)
print("\nScore for linear regression training set:\n", acc_train)

from sklearn.metrics import mean_squared_error
student_predictions = lin_regr.predict(X_train_prepared)
lin_mse = mean_squared_error(y_train, student_predictions)
lin_rmse = np.sqrt(lin_mse)
print("\nRMSE for linear regression:\n", lin_rmse)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_regr, X_train_prepared, y_train,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)

print("\nCross validation for linear regression:\n")
def display_scores(scores):
    print("\tScores:\n", scores)
    print("\tMean:\n", scores.mean())
    print("\tStandard deviation:\n", scores.std())

display_scores(lin_rmse_scores)


from sklearn.tree import DecisionTreeRegressor
tree_regr = DecisionTreeRegressor()
tree_regr.fit(X_train_prepared, y_train)

acc_train_tree = tree_regr.score(X_train_prepared, y_train)
print("\nScore for tree regression training set:\n", acc_train_tree)

student_predictions_tree = tree_regr.predict(X_train_prepared)
tree_mse = mean_squared_error(y_train, student_predictions_tree)
tree_rmse = np.sqrt(tree_mse)
print("\nRMSE for tree regression:\n", tree_rmse)

scores = cross_val_score(tree_regr, X_train_prepared, y_train,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("\nCross validation for tree regression:\n")
display_scores(tree_rmse_scores)


from sklearn.ensemble import RandomForestRegressor
forest_regr = RandomForestRegressor()
forest_regr.fit(X_train_prepared, y_train)

acc_train_forest = forest_regr.score(X_train_prepared, y_train)
print("\nScore for forest regression training set:\n", acc_train_forest)

student_predictions_forest = forest_regr.predict(X_train_prepared)
forest_mse = mean_squared_error(y_train, student_predictions_forest)
forest_rmse = np.sqrt(forest_mse)
print("\nRMSE for forest regression:\n", forest_rmse)

scores = cross_val_score(forest_regr, X_train_prepared, y_train,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

print("\nCross validation for forest regression:\n")
display_scores(forest_rmse_scores)


from sklearn.model_selection import GridSearchCV
param_grid = [
                {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
            ]
grid_search = GridSearchCV(forest_regr, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train_prepared, y_train)

print("\nGrid search best estimator for random forest:\n", grid_search.best_estimator_)

feature_importances = grid_search.best_estimator_.feature_importances_
attributes = ["sex", "address", "famsize", "Pstatus", "Mother edu", "Father edu", "studytime", "failures", "schoolsup",
             "paid", "higher", "absences", "G1", "G2"]
print(sorted(zip(feature_importances, attributes), reverse=True))



final_model = grid_search.best_estimator_

X_test = test_set.drop("G3", axis=1)
y_test = test_set["G3"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("\nFINAL RMSE for test data:\n", final_rmse)


from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
get_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
print("A 95% confidence interval:\n", get_interval)


# Drawing and plotting model
# plot = "failures"
# plt.scatter(data[plot], data["G3"])
# plt.legend(loc=4)
# plt.xlabel(plot)
# plt.ylabel("Final Grade")
# plt.show()