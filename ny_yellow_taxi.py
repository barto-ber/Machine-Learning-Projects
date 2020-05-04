import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
pd.options.display.width = 0
pd.options.display.max_rows = None

data = dd.read_csv('2018_Yellow_Taxi_Trip_Data.csv')

data = data[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID',
             'payment_type', 'fare_amount']]

# print(data.head())
# print(data.info())

'''CHECK FOR NaNs (generally and in detail)'''
# with ProgressBar():
#     missing_values_all = data.isnull().sum().sum().compute()
# print("\nHow many NaNs in dataset?\n", missing_values_all)
#
# missing_values = data.isnull().sum()
# missing_count = (missing_values / data.index.size * 100)
# with ProgressBar():
#     missing_count_pct = missing_count.compute()
# print("\nWhere are the NaNs:\n", missing_count_pct)

'''How many rows in dataset'''
# with ProgressBar():
#     rows_sum = data.index.size.compute()
# print("So many rows in dataset:\n", rows_sum)

'''Throwing out unknown zones 264 & 265 and zone 1 (Newark Airport)'''
data = data[data['PULocationID'] != 1]
data = data[data['DOLocationID'] != 1]
data = data[data['PULocationID'] < 264]
data = data[data['DOLocationID'] < 264]
data = data[data['RatecodeID'] == 1]
data = data[data['fare_amount'] > 0]
data = data[data['trip_distance'] > 0]
data['trip_distance'] = (data['trip_distance'] * 1.609344).round(decimals=2) # getting KM

'''Converting to datetime'''
data['tpep_pickup_datetime'] = dd.to_datetime(data['tpep_pickup_datetime'])
data['tpep_dropoff_datetime'] = dd.to_datetime(data['tpep_dropoff_datetime'])
data['trip_time'] = data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']

print(data.head())
print(data.dtypes)
# with ProgressBar():
#     min_date = data['tpep_pickup_datetime'].min().compute()
# print(min_date)

'''Check for unknown zones etc.'''
# with ProgressBar():
    # unknown_zones = (data['PULocationID'] >= 264).sum().compute()
    # standard_rate = (data['RatecodeID'] > 1).sum().compute()
    # no_charge = (data['payment_type'] >= 3).sum().compute()
    # no_charge = data[(data['payment_type'] >= 3)]
    # zero_charge = data[(data['fare_amount'] == 3)]
# print("\nHow many unknown zones in dataset?\n", unknown_zones)
# print("\nHow many not standard trip rates in dataset?\n", standard_rate)
# print("\nHow many not charged etc. in dataset?\n", no_charge.head())
# print("\nHow many with zero-close charges in dataset?\n", zero_charge.head())

'''How many rows in dataset'''
# with ProgressBar():
#     rows_sum = data.index.size.compute()
# print("So many rows in dataset:\n", rows_sum)



