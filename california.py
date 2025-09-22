
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler,FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer,make_column_selector, make_column_transformer
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint


# read and show the dataset
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR /"datasets"/"housing.csv"
housing = pd.read_csv(file_path)

"""
print(housing.head()) # the top five rows
print(housing.info()) # to get a quick description of the data
print(housing["ocean_proximity"].value_counts())
print(housing.describe()) # summary of the numerical attribute

"""

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


# Visualizing Geographical Data
"""

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100, label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10, 7))
plt.show()

"""


# Look for Correlations
"""

# option 1
corr_matrix = housing.select_dtypes(include=["number"]).corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


# option 2
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()


housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1, grid=True)
plt.show()

"""


# Experiment with Attribute Combinations
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
corr_matrix = housing.select_dtypes(include=["number"]).corr()

# Prepare the Data for Machine Learning Algorithms
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# Clean the data
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
X = imputer.transform(housing_num) # output of imputer.transform(housing_num) is a NumPy array:  
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


# Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot.toarray())



# Feature Scaling and Transformation

# option 1  Min-max scaling (many people call this normalization)
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# option 2  
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# Custom Transformers
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))


# Transformation Pipelines
num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")),("standardize", StandardScaler())])


num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households" "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([("num", num_pipeline, num_attribs),("cat", cat_pipeline, cat_attribs),])

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)

# print(housing_prepared.shape)
# print(preprocessing.get_feature_names_out())


# Train and Evaluate on the Training Set

# Linear regression
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
# print(lin_rmse)

# Decision tree
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
# print(tree_rmse)


# Better Evaluation Using Cross-Validation

"""

WARNING
Scikit-Learn’s cross-validation features expect a utility function (greater is better) rather
than a cost function (lower is better), so the scoring function is actually the opposite of the
RMSE. It’s a negative value, so you need to switch the sign of the output to get the RMSE
scores

"""
tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(tree_rmses).describe())


# Random Forest
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(forest_rmses).describe())


# Fine-Tune Your Model

# Grid Search


full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

param_grid = [
    {'random_forest__max_features': [4, 6, 8]},
    {'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)

# print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
# print(cv_res.head())



# Randomized Search
param_distribs = {
        'random_forest__max_features': randint(low=2, high=20)
}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42
)

rnd_search.fit(housing, housing_labels)
# print(rnd_search.best_params_)

# Ensemble Methods

# Analyzing the Best Models and Their Errors
final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
# print(feature_importances.round(2))


print(sorted(zip(feature_importances,final_model["preprocessing"].get_feature_names_out()),reverse=True))