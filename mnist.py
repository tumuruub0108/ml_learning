from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "datasets" / "mnist_train.csv"
mnist = pd.read_csv(file_path)

X = np.array(mnist.drop(['label'], axis=1))
y = np.array(mnist['label'])

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()

some_digit = X[0]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Training a Binary Classifier
# Binary target: is digit 5?
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Train SGD classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Predict
# print("Prediction for first digit:", sgd_clf.predict([some_digit]))
# print("Decision function value:", sgd_clf.decision_function([some_digit]))

# Measuring Accuracy Using Cross-Validation

# let’s look at a dummy classifier
dummy_clf = DummyClassifier()
dummy_clf.fit(X_train,y_train_5)
# print(any(dummy_clf.predict(X_train))) # prints False: no 5s detected
# print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))



# Confusion Matrices
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)