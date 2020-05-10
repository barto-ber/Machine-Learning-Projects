from datetime import datetime
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
pd.options.display.width = 0
pd.options.display.max_rows = None
pd.options.display.float_format = "{:.2f}".format

time_now = datetime.now()

'''Reading data from CSV'''
print(time_now, "--- Reading data ---\n")
data = pd.read_csv('D:\Coding_data\yellow_tripdata_2019-01.csv')

data = data[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID',
             'payment_type', 'fare_amount']]

# print(data.head())
# print(data.describe())

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

'''Throwing out unknown zones 264 & 265 and zone 1 (Newark Airport)'''
print("--- Throwing out bad data points ---\n")
data = data[data['PULocationID'] != 1]
data = data[data['DOLocationID'] != 1]
data = data[data['PULocationID'] < 264]
data = data[data['DOLocationID'] < 264]
data = data[data['RatecodeID'] == 1]
data = data[data['fare_amount'] > 0]
data['trip_distance'] = (data['trip_distance'] * 1.609344).round(decimals=2) # getting KM
data = data[data['trip_distance'] >= 0.5]
data = data[data['trip_distance'] < 100]
data = data[data['fare_amount'] < 150]
data = (data[data['fare_amount'] >= 2.5])
data = data[data['payment_type'] <= 2]

# print(data.describe())

'''Converting to datetime'''
print("--- Converting to datetime ---\n")
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])
data['trip_time'] = data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']

'''Copying clean data to CSV'''
# print("--- Copying clean data to CSV ---\n")
# data.to_csv("yellow_tripdata_2019-1_V1.csv", index=False)

'''Reading clean data'''
# print("--- Reading clean data ---\n")
# data = pd.read_csv('yellow_tripdata_2019-1_V1.csv')
# print(data.head())
# print(data.dtypes)
# with ProgressBar():
#     min_date = data['tpep_pickup_datetime'].min().compute()
# print(min_date)

'''Converting datetime to unix'''
print("--- Converting datetime to unix time ---\n")
data_unix = data.copy()
data_unix['tpep_pickup_datetime'] = (data_unix['tpep_pickup_datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
data_unix['tpep_dropoff_datetime'] = (data_unix['tpep_dropoff_datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

print("--- Adding additional attribute combinations ---\n")
data_unix['trip_time'] = data_unix['tpep_dropoff_datetime'] - data_unix['tpep_pickup_datetime']
data_unix['distance/time'] = data_unix['trip_distance'] / data_unix['trip_time'] # lin correlations close zero
# print("Unix time data head:\n", data_unix.head())

'''Creating train and test set'''
print("--- Creating train and test set ---\n")
train_set, test_set = train_test_split(data_unix, test_size=0.3, random_state=42)

'''Looking for correlations'''
# print("--- Checking for linear pearsons correlations ---\n")
# corr_matrix = train_set.corr()
# check_fare_amount = corr_matrix['fare_amount'].sort_values(ascending=False)
# print("\nCheck fare amount correlations:\n", check_fare_amount)

'''Throwing out last irrelevant features'''
print("--- Throwing out irrelevant features ---\n")
irrelevant_features = ['RatecodeID', 'payment_type']
data_unix.drop(irrelevant_features, inplace=True, axis=1)

'''Last check for infinity and NaN and setting inf as Na'''
# missing_values_all = data_unix.isnull().sum().sum()
# print("\nHow many NaNs in dataset?\n", missing_values_all)

# missing_values = data_unix.isnull().sum()
# missing_count = (missing_values / data_unix.index.size * 100)
# print("\nWhere are the NaNs:\n", missing_count)

pd.set_option('use_inf_as_na', True)
data_unix = data_unix.replace([np.inf, -np.inf], 0).dropna(subset=data_unix.columns, how="all")
data_unix.dropna()
# print("Pandas check if all elements are finite?:\n", data_unix.notnull().values.all())
# print("Numpy check if all elements are finite?:\n", np.isfinite(data_unix).sum())
#
# print("Pandas check if any element is Na (not available)?:\n", data_unix.isnull().values.all())
# print("Numpy check if any element is Na (not available)?:\n", np.isnan(data_unix).all())

# print("Are there any NaN???:\n", data_unix.isna().sum())

'''Split data'''
print("--- Splitting data ---\n")

predict = 'fare_amount'

X = data_unix.drop([predict], axis=1)
y = data_unix[predict]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''Scaling and transforming data'''
print("--- Scaling and transforming data ---\n")
numerical_X_train = X_train.drop(['PULocationID', 'DOLocationID'], axis=1)
num_attribs_X_train = list(numerical_X_train)
cat_attribs_X_train = ['PULocationID', 'DOLocationID']

numerical_X_test = X_test.drop(['PULocationID', 'DOLocationID'], axis=1)
cat_attribs_X_test = ['PULocationID', 'DOLocationID']

scale_transform = ColumnTransformer([
                            ('scaler', StandardScaler(), num_attribs_X_train),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs_X_train)
                        ])

X_train_prepared = scale_transform.fit_transform(X_train)
X_test_prepared = scale_transform.transform(X_test)

'''Linear regression'''
# print("--- Calculating Linear Regression ---\n")
# lin_reg = LinearRegression()
#
# print("\n\tCross validation with RMSE for Linear Regression:")
# lin_reg_scores_nMSE = cross_val_score(lin_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
# lin_reg_rmse_scores = np.sqrt(-lin_reg_scores_nMSE)
# print("\nScores RMSE:\n", lin_reg_rmse_scores)
# print("\nMean score RMSE:\n", lin_reg_rmse_scores.mean())
# print("\nStandard deviation:\n", lin_reg_rmse_scores.std())
#
# print("\n\tCross validation with MAE for Linear Regression:")
# lin_reg_scores_MAE = cross_val_score(lin_reg, X_train_prepared, y_train, scoring="neg_mean_absolute_error", cv=10, n_jobs=-1)
# print("\nScores MAE:\n", lin_reg_scores_MAE)
# print("\nMean score MAE:\n", lin_reg_scores_MAE.mean())
# print("\nStandard deviation:\n", lin_reg_scores_MAE.std())
#
# print("\n\tCross validation with R^2 for Linear Regression:")
# lin_reg_scores_r2 = cross_val_score(lin_reg, X_train_prepared, y_train, scoring="r2", cv=10, n_jobs=-1)
# print("\nScores R^2:\n", lin_reg_scores_r2)
# print("\nMean score R^2:\n", lin_reg_scores_r2.mean())
# print("\nStandard deviation:\n", lin_reg_scores_r2.std())

# lin_reg.fit(X_train_prepared, y_train)
# predictions = lin_reg.predict(X_test_prepared)
#
# score_r2_lin_reg = r2_score(y_test, predictions)
# print("\nR^2 of Linear Regression model:\n", score_r2_lin_reg)
#
# lin_reg_mse = mean_squared_error(y_test, predictions)
# lin_reg_rmse = np.sqrt(lin_reg_mse)
# print("\nMSE of Linear Regression model:\n", lin_reg_mse)
# print("\nRMSE of Linear Regression model:\n", lin_reg_rmse)
#
# lin_reg_mae = mean_absolute_error(y_test, predictions)
# print("\nMAE of Linear Regression model:\n", lin_reg_mae)
#
# lin_reg_confidence = 0.95
# lin_reg_squared_errors = (predictions - y_test) ** 2
# interv_95 = np.sqrt(stats.t.interval(lin_reg_confidence, len(lin_reg_squared_errors) - 1,
#                          loc=lin_reg_squared_errors.mean(),
#                          scale=stats.sem(lin_reg_squared_errors)))
# print("\n95% confidence interval for the generalization error of Linear Regression model:\n", interv_95)

'''Plot outputs Linear Regression - doesnt work!!!'''
# plt.scatter(X_test_prepared[2], y_test,  color='black', alpha=0.1)
# plt.plot(X_test_prepared[2], predictions, color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()

'''Polynomial regression'''
# print("--- Calculating Polynomial Regression ---\n")
# pipe_poly_reg = make_pipeline(
#     PolynomialFeatures(degree=2, include_bias=False),
#     LinearRegression()
# )

# param_grid_poly_reg = {'polynomialfeatures__degree': [1, 2, 3]}
#
# grid_poly_reg = GridSearchCV(pipe_poly_reg, param_grid=param_grid_poly_reg, cv=5, n_jobs=-1)
# grid_poly_reg.fit(X_train_prepared, y_train)
#
# print("Best parameters for Polynomial Regression:\n", grid_poly_reg.best_params_)
# print("Best cross-validation score for Polynomial Regression:\n", grid_poly_reg.best_score_)

# pipe_poly_reg.fit(X_train_prepared, y_train)
# predictions = pipe_poly_reg.predict(X_test_prepared)
#
# score_r2_poly_reg = r2_score(y_test, predictions)
# print("\nR^2 of Polynomial Regression model:\n", score_r2_poly_reg)
#
# poly_reg_mse = mean_squared_error(y_test, predictions)
# poly_reg_rmse = np.sqrt(poly_reg_mse)
# print("\nMSE of Polynomial Regression model:\n", poly_reg_mse)
# print("\nRMSE of Polynomial Regression model:\n", poly_reg_rmse)
#
# poly_reg_mae = mean_absolute_error(y_test, predictions)
# print("\nMAE of Polynomial Regression model:\n", poly_reg_mae)
#
# poly_reg_confidence = 0.95
# poly_reg_squared_errors = (predictions - y_test) ** 2
# interv_95 = np.sqrt(stats.t.interval(poly_reg_confidence, len(poly_reg_squared_errors) - 1,
#                          loc=poly_reg_squared_errors.mean(),
#                          scale=stats.sem(poly_reg_squared_errors)))
# print("\n95% confidence interval for the generalization error of Polynomial Regression model:\n", interv_95)

'''Polynomial Ridge Regression'''
# print("--- Calculating Polynomial Ridge Regression ---\n")
# pipe_poly_ridge = make_pipeline(
#     PolynomialFeatures(degree=2, include_bias=False),
#     Ridge()
# )

# param_grid_poly_ridge = {'polynomialfeatures__degree': [1, 2, 3],
#                          'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
#
# grid_poly_ridge = GridSearchCV(pipe_poly_ridge, param_grid=param_grid_poly_ridge, cv=5, n_jobs=-1)
# grid_poly_ridge.fit(X_train_prepared, y_train)
#
# print("Best parameters for Polynomial Ridge Regression:\n", grid_poly_ridge.best_params_)
# print("Best cross-validation score for Polynomial Ridge Regression:\n", grid_poly_ridge.best_score_)
#
# plt.matshow(grid_poly_ridge.cv_results_['mean_test_score'].reshape(3, -1), vmin=0, cmap="viridis")
# plt.xlabel("ridge__alpha")
# plt.ylabel("polynomialfeatures__degree")
# plt.xticks(range(len(param_grid_poly_ridge['ridge__alpha'])), param_grid_poly_ridge['ridge__alpha'])
# plt.yticks(range(len(param_grid_poly_ridge['polynomialfeatures__degree'])), param_grid_poly_ridge['polynomialfeatures__degree'])
# plt.colorbar()

# pipe_poly_ridge.fit(X_train_prepared, y_train)
# predictions = pipe_poly_ridge.predict(X_test_prepared)
#
# score_r2_poly_ridge = r2_score(y_test, predictions)
# print("\nR^2 of Polynomial Ridge Regression model:\n", score_r2_poly_ridge)
#
# poly_ridge_mse = mean_squared_error(y_test, predictions)
# poly_ridge_rmse = np.sqrt(poly_ridge_mse)
# print("\nMSE of Polynomial Ridge Regression model:\n", poly_ridge_mse)
# print("\nRMSE of Polynomial Ridge Regression model:\n", poly_ridge_rmse)
#
# poly_ridge_mae = mean_absolute_error(y_test, predictions)
# print("\nMAE of Polynomial Ridge Regression model:\n", poly_ridge_mae)
#
# poly_ridge_confidence = 0.95
# poly_ridge_squared_errors = (predictions - y_test) ** 2
# interv_95 = np.sqrt(stats.t.interval(poly_ridge_confidence, len(poly_ridge_squared_errors) - 1,
#                          loc=poly_ridge_squared_errors.mean(),
#                          scale=stats.sem(poly_ridge_squared_errors)))
# print("\n95% confidence interval for the generalization error of Polynomial Ridge Regression model:\n", interv_95)

'''Random Forest'''
# print("--- Calculating Random Forest ---")
# forest = RandomForestRegressor(n_jobs=-1, random_state=42)

# print("\n\tCross validation with RMSE for Random Forest:")
# forest_scores_nMSE = cross_val_score(forest, X_train, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
# forest_rmse_scores = np.sqrt(-forest_scores_nMSE)
# print("\nScores RMSE:\n", forest_rmse_scores)
# print("\nMean score RMSE:\n", forest_rmse_scores.mean())
# print("\nStandard deviation:\n", forest_rmse_scores.std())
#
# print("\n\tCross validation with MAE for Random Forest:")
# forest_scores_MAE = cross_val_score(forest, X_train, y_train, scoring="neg_mean_absolute_error", cv=10, n_jobs=-1)
# print("\nScores MAE:\n", forest_scores_MAE)
# print("\nMean score MAE:\n", forest_scores_MAE.mean())
# print("\nStandard deviation:\n", forest_scores_MAE.std())
#
# print("\n\tCross validation with R^2 for Random Forest:")
# forest_scores_r2 = cross_val_score(forest, X_train, y_train, scoring="r2", cv=10, n_jobs=-1)
# print("\nScores R^2:\n", forest_scores_r2)
# print("\nMean score R^2:\n", forest_scores_r2.mean())
# print("\nStandard deviation:\n", forest_scores_r2.std())

# forest.fit(X_train, y_train)
# predictions = forest.predict(X_test)
#
# score_r2_forest = r2_score(y_test, predictions)
# print("\nR^2 of Random Forest model:\n", score_r2_forest)
#
# forest_mse = mean_squared_error(y_test, predictions)
# forest_rmse = np.sqrt(forest_mse)
# print("\nMSE of Random Forest model:\n", forest_mse)
# print("\nRMSE of Random Forest model:\n", forest_rmse)
#
# forest_mae = mean_absolute_error(y_test, predictions)
# print("\nMAE of Random Forest model:\n", forest_mae)
#
# forest_confidence = 0.95
# forest_squared_errors = (predictions - y_test) ** 2
# interv_95 = np.sqrt(stats.t.interval(forest_confidence, len(forest_squared_errors) - 1,
#                          loc=forest_squared_errors.mean(),
#                          scale=stats.sem(forest_squared_errors)))
# print("\n95% confidence interval for the generalization error of Random Forest model:\n", interv_95)

'''XGBOOST'''
print("--- Calculating XGB Reg ---")
xgb_reg = XGBRegressor()

# print("\n\tCross validation with RMSE for XGB:")
# xgb_scores_nMSE = cross_val_score(xgb_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
# xgb_rmse_scores = np.sqrt(-xgb_scores_nMSE)
# print("\nScores RMSE:\n", xgb_rmse_scores)
# print("\nMean score RMSE:\n", xgb_rmse_scores.mean())
# print("\nStandard deviation:\n", xgb_rmse_scores.std())
#
# print("\n\tCross validation with MAE for XGB:")
# xgb_scores_MAE = cross_val_score(xgb_reg, X_train, y_train, scoring="neg_mean_absolute_error", cv=10, n_jobs=-1)
# print("\nScores MAE:\n", xgb_scores_MAE)
# print("\nMean score MAE:\n", xgb_scores_MAE.mean())
# print("\nStandard deviation:\n", xgb_scores_MAE.std())
#
# print("\n\tCross validation with R^2 for Random Forest:")
# xgb_scores_r2 = cross_val_score(xgb_reg, X_train, y_train, scoring="r2", cv=10, n_jobs=-1)
# print("\nScores R^2:\n", xgb_scores_r2)
# print("\nMean score R^2:\n", xgb_scores_r2.mean())
# print("\nStandard deviation:\n", xgb_scores_r2.std())

xgb_reg.fit(X_train, y_train)
predictions = xgb_reg.predict(X_test)

score_r2_xgb = r2_score(y_test, predictions)
print("\nR^2 of XGB model:\n", score_r2_xgb)

xgb_mse = mean_squared_error(y_test, predictions)
xgb_rmse = np.sqrt(xgb_mse)
print("\nMSE of XGB model:\n", xgb_mse)
print("\nRMSE of XGB model:\n", xgb_rmse)

xgb_mae = mean_absolute_error(y_test, predictions)
print("\nMAE of XGB model:\n", xgb_mae)

xgb_confidence = 0.95
xgb_squared_errors = (predictions - y_test) ** 2
interv_95 = np.sqrt(stats.t.interval(xgb_confidence, len(xgb_squared_errors) - 1,
                         loc=xgb_squared_errors.mean(),
                         scale=stats.sem(xgb_squared_errors)))
print("\n95% confidence interval for the generalization error of XGB model:\n", interv_95)

