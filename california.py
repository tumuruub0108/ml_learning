
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

housing.hist(bins=50, figsize=(12, 8)) # to plot a histogram for each numerical attribute
plt.show()