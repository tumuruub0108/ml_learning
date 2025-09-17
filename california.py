
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression

# read and show the dataset
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR /"datasets"/"housing.csv"
housing = pd.read_csv(file_path)


# print(housing.head()) # the top five rows
# print(housing.info()) # to get a quick description of the data
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe()) # summary of the numerical attribute
"""

housing.hist(bins=50, figsize=(12, 8)) # to plot a histogram for each numerical attribute
plt.show()

"""

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])

"""

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()

"""

strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)   
    
housing = strat_train_set.copy()


housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
plt.show()