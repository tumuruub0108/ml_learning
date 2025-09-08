
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
df_house = pd.read_csv(file_path)


train_set, test_set = train_test_split(df_house, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="median")
housing_num = df_house.select_dtypes(include=[np.number])
imputer.fit(housing_num)


X = imputer.transform(housing_num)
df_housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)

# Handling Text and Categorical Attributes
data = pd.DataFrame({"ocean_proximity": ["NEAR BAY", "INLAND", "NEAR OCEAN", "INLAND"]})

cat_encoder = OneHotEncoder()
X = df_house[["ocean_proximity"]]
X_encoded = cat_encoder.fit_transform(X)

X_encoded_df = pd.DataFrame(
    X_encoded.toarray(),
    columns=cat_encoder.get_feature_names_out(["ocean_proximity"]),
    index=df_house.index  # now X_encoded has same number of rows as df_house
)

df_house = df_house.drop("ocean_proximity", axis=1)
df_house = pd.concat([df_house, X_encoded_df], axis=1)

# Feature Scaling and Transformation

# Min-max scaling
min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
# print(housing_num_min_max_scaled)

# Standardization
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
# print(housing_num_std_scaled)


age_simil_35 = rbf_kernel(df_house[["housing_median_age"]],[[35]], gamma=0.1)
print(age_simil_35)
