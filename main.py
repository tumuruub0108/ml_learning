

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler 

# reading data 
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("./datataset/magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)   

"""

for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color='blue', label="gamma", alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color='red', label="hadron", alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
    
"""

# Train, Validation, test datasets
# train(60%), Validation(20%), Test(20%)
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*(len(df)))])


"""
what does "imbalanced" mean?
A dataset is imbalanced if one class(e.g gamma = 1) has much more data than the other(e.g hadron = 0)

If each class is within ~40–60% range, it’s balanced.
If one class is <30% of the data, it’s imbalanced.

Example balanced: gamma:8500, hadron:8700
Example imbalanced: gamma: 18000, hadron:5000 




Balancing is important because:

It prevents the model from ignoring the minority class.
It ensures both gamma (1) and hadron (0) are learned properly.
Otherwise, you risk building a “biased” model that looks accurate but is useless in practice.'


What is minority class
The minority class = the class that has fewer samples in your datatset
The minority class = the class that has more samples

✅ Example 1 (balanced)
Gamma (1): 8500 samples
Hadron (0): 8700 samples

Both are about the same.
No minority class here -> dataset is balanced

✅ Example 2 (imbalanced)
Gamma (1): 3000 samples
Hadron (0): 17000 samples

Gamma (1) = minority class (only 15%)
Hadron (0) = majority class (85%)

🚀 Why it matters

Machine learning models tend to favor the majority class, because it dominates the data.

The minority class is usually the one you care about detecting (e.g., in fraud detection, fraud cases are minority but very important).

In your gamma/hadron dataset, if one side has far fewer samples, that’s your minority class.


✅ Why balanced datasets help

Fair learning:
The model gets enough examples of each class, so it learns both patterns.

Better metrics:
Balanced data improves precision, recall, F1-score, not just accuracy.
(Especially important if detecting gamma events is the real goal.)

Avoid bias:
Without balance, the model is biased toward the majority class.

"""
print(len(train[train["class"] == 1])) # gamma result = 7384  
print(len(train[train["class"] == 0])) # hadron resutl = 4028


def scale_dataset(df, oversample=False):
    # df.cols[:-1] → df–ийн сүүлийн баганаас бусад бүх баганы нэрийг авна.
    # Эндээс харахад сүүлийн багана нь target (label), бусад баганууд нь feature гэж үзэж байна.s
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler()
        X,y = ros.fit_resample(X, y)
    
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)