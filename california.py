
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# read and show the dataset
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR /"datasets"/"housing.csv"
df_house = pd.read_csv(file_path)

print(df_house.head(5))