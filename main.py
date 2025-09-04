

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


"""
sklearn.metrics
🔎 What are “metrics” in machine learning?
Metrics are measurements used to evaluate how well a machine learning model performs.
They tell you how good your predictions are compared to the true labels.


1️⃣ Types of metrics
For classification:

Accuracy: Fraction of correct predictions
Precision: Of all predicted as class X, how many were actually X
Recall: Of all actual X, how many did we predict correctly
F1-score: Harmonic mean of precision and recall
Confusion Matrix: Table of true positives, false positives, etc.

For regression (numeric prediction):

Mean Squared Error (MSE): Average squared difference between predicted and actual values
Mean Absolute Error (MAE): Average absolute difference
R² score: How much variance is explained by the mode



"""

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
train_df, valid_df, test_df = train, valid, test

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
print("train =", len(train[train["class"] == 1])) # 
print("train =", len(train[train["class"] == 0])) # 

"""
    # Example DataFrame
    df = pd.DataFrame({
    "f1": [1, 2, 3],
    "f2": [4, 5, 6],
    "class": ["g", "h", "g"]
})

print(df)

   f1  f2 class
0   1   4     g
1   2   5     h
2   3   6     g

Step 1: Get all column names
print(df.columns)
ndex(['f1', 'f2', 'class'], dtype='object')

Step 2: Pick the last column name
print(df.columns[-1])
'class'


Step 3: Select that column
y = df[df.columns[-1]]
print(y)

0    g
1    h
2    g
Name: class, dtype: object

Step 4: Convert to NumPy array
y = df[df.columns[-1]].values
print(y)

array(['g', 'h', 'g'], dtype=object)


"""

def scale_dataset(df, oversample=False):
    # df.cols[:-1] → df–ийн сүүлийн баганаас бусад бүх баганы нэрийг авна.
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler()
        X,y = ros.fit_resample(X, y)
    
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y

train, X_train, y_train = scale_dataset(train_df, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid_df, oversample=False)
test, X_test, y_test = scale_dataset(test_df, oversample=False)
"""

print("y_train=", len(y_train))
print("y_train==1", sum(y_train == 1))
print("y_train==0", sum(y_train == 0))

"""

# K-Nearest Neighbors

"""
KNN нь машин суралцах (ML) алгоритм юм.
Төрөл: Supervised learning (classification болон regression-д хэрэглэгддэг)
Үндсэн санаа: Шинэ өгөгдлийн цэгийг хамгийн ойрын K хөршүүдийн мэдээллээр таамаглах.


1️⃣ Хэрхэн ажилладаг вэ?

K-г сонгоно (жишээ нь K = 3)
Шинэ өгөгдлийн цэгт хамгийн ойр байгаа K хөршийг олно
Classification-д: Хөршүүдийн дийлэнх классыг шинж чанарын цэгийн хариулт болгоно
Regression-д: Хөршүүдийн утгын дундажийг хариулт болгоно


🔎 Жишээ (Classification)

Таны өгөгдөл: Gamma (1) ба Hadron (0)

K = 5
Шинэ өгөгдөл оруулахад:
    5 хамгийн ойрын хөршийг олно
    Хөршүүдийн дийлэнх нь Gamma бол → шинж цэгийг Gamma гэж таамаглана
    Хөршүүдийн дийлэнх нь Hadron бол → Hadron гэж таамаглана


🔎 Жишээ (Regression)

Хөршүүдийн үнэлгээ: [2.0, 3.0, 3.5, 2.5, 4.0]
K = 5 → шинж цэгийн утга = (2+3+3.5+2.5+4)/5 = 3.0


"""
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_prediction = knn_model.predict(X_test)
print("y_prediction=",y_prediction)
print("y_test=", y_test)
print(classification_report(y_test,y_prediction))


# Naive Bayes
