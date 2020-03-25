import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.display.width = 0
pd.options.display.max_rows = None
pd.options.display.precision = 5


def read_housing():
    dtypes = {
        "regio1": "category",
        "firingTypes": "category",
        "heatingType": "category",
        "petsAllowed": "category",
        "typeOfFlat": "category"
    }
    data = pd.read_csv("immo_data.csv", dtype=dtypes)
    return data


def drop_columns():
    data = read_housing()
    less_columns = ["telekomTvOffer", "serviceCharge", "telekomHybridUploadSpeed", "newlyConst", "picturecount", "pricetrend",
                    "geo_bln", "yearConstructedRange", "interiorQual", "street", "baseRentRange", "thermalChar", "regio2",
                    "description", "facilities", "energyEfficiencyClass", "electricityBasePrice", "electricityKwhPrice",
                    "telekomUploadSpeed", "noParkSpaces", "garden", "livingSpaceRange", "heatingCosts", "noRoomsRange",
                    "cellar", "lift"]
    data.drop(less_columns, inplace=True, axis=1)
    data = data[data['totalRent'] < 3000]
    data = data[data['baseRent'] < 3000]
    data = data[data['noRooms'] < 8]
    data = data[data['floor'] <= 20]
    data = data[data['numberOfFloors'] <= 20]
    data = data[data['livingSpace'] <= 500]
    data = data[data['yearConstructed'] <= 2020]
    data = data[data['yearConstructed'] >= 1850]
    data = data[data['lastRefurbish'] <= 2020]
    return data


def data_geo_num():
    data = drop_columns()
    # data['regio1_num'] = pd.factorize(data.regio1)[0]
    regio1_to_nums = {'regio1':
        {'Berlin': 0, 'Hamburg': 1, 'Bayern': 2, 'Hessen': 3,
         'Baden_Württemberg': 4, 'Nordrhein_Westfalen': 5,
         'Rheinland_Pfalz': 6, 'Niedersachsen': 7, 'Sachsen': 8,
         'Mecklenburg_Vorpommern': 9, 'Sachsen_Anhalt': 10,
         'Saarland': 11, 'Bremen': 12, 'Schleswig_Holstein': 13,
         'Thüringen': 14, 'Brandenburg': 15}
    }
    data.replace(regio1_to_nums, inplace=True)
    print("\nData with Lands factorized:\n", data.head())
    print("\nCheck Values:\n", data['regio1'].unique())
    return data


def data_median_rent():
    data_with_median_rent = data_geo_num()
    m = data_with_median_rent.groupby('regio1')['baseRent']
    data_with_median_rent['median_base_rent'] = m.transform(np.median)
    data_with_median_rent['rooms_per_livingspace'] = data_with_median_rent['noRooms'] / data_with_median_rent['livingSpace']
    print("\nMedian base rent in Lands:\n", data_with_median_rent.head())
    return data_with_median_rent


def check_data():
    data = data_median_rent()
    print(data[:50])
    print(data.info())
    # print(data.describe())
    # print("\nLets see PLZ:\n", data[
    #                                 (data['geo_plz'] <= 10000)].head())
    # print("\nCheck Values:\n", data[data['geo_plz'].count_values)
    # print("\nChecking how many values in PLZ:\n", data.groupby('geo_plz').size())
check_data()

def data_histogram():
    data = data_geo_num()
    data.hist(bins=50)
    plt.show()


def data_2d_hist():
    data = drop_columns()
    x, y = data['baseRent'], data['yearConstructed']
    plt.hist2d(x, y, bins=50, cmap='Blues')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    plt.show()


def plz_hist():
    data = data_geo_num()
    data['regio1_category'] = pd.cut(data['regio1'],
                                  bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, np.inf],
                                  labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    data['regio1_category'].hist()
    plt.show()


def land_scatter():
    data = data_median_rent()
    data.plot(kind="scatter", x="regio1", y='median_base_rent')
    plt.show()


# Creating a Test Set
from sklearn.model_selection import StratifiedShuffleSplit

def create_test_set():
    data = data_median_rent()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["regio1"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
        check_test = strat_test_set["regio1"].value_counts() / len(strat_test_set)
        print(check_test)
        # print("\nCheck strat train:\n", strat_train_set.info()) # Result NONE, if checking whole dtrat training mostly NANs. WHY???
        return strat_train_set, strat_test_set
create_test_set()

def data_labels():
    data, strat_train_set = create_test_set()
    data = strat_train_set.drop("median_base_rent", axis=1)
    data_labels = strat_train_set["median_base_rent"].copy()
    print("\nData labels info:\n", data_labels)
    return data, data_labels


def looking_for_correlations():
    data = data_median_rent()
    corr_matrix = data.corr()
    print("\nCorrelations:\n", corr_matrix["baseRent"].sort_values(ascending=False))


from pandas.plotting import scatter_matrix
def plot_correlations():
    data = data_median_rent()
    attributes = ['baseRent', 'livingSpace', 'noRooms', 'regio1']
    scatter_matrix(data[attributes], figsize=(12, 8))
    plt.show()


def most_promising_corr_scater():
    data = data_median_rent()
    data.plot(kind="scatter", x="lastRefurbish", y='baseRent', alpha=0.1)
    plt.show()









