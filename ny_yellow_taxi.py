import pandas as pd
import dask.dataframe as dd
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

# dtypes = {"sex": "category",
#           "address": "category",
#           "famsize": "category",
#           "Pstatus": "category",
#           "schoolsup": "category",
#           "paid": "category",
#           "higher": "category"}

data = dd.read_csv('2018_Yellow_Taxi_Trip_Data.csv')
# data = data[["sex", "address", "famsize", "Pstatus", "Mother edu", "Father edu", "studytime", "failures", "schoolsup",
#              "paid", "higher", "absences", "G1", "G2", "G3"]]
print(data.head())
print(data.info())
# print("\nHow many NaN in dataset?\n", data.isnull().sum().sum())