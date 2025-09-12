from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]


# Training a Binary Classifier
y_train_5 = (y_train == '5') # True for all 5s, False for all other digits
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# print(sgd_clf.predict([some_digit]))

# Performance Measures


# Measuring Accuracy Using Cross-Validation
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")) # the result is (ratio of correct predictions)

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
# print(any(dummy_clf.predict(X_train))) # prints False: no 5s detected

# print(cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# Confusion Matrices
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
# print(cm)

y_train_perfect_predictions = y_train_5
# print(confusion_matrix(y_train_5, y_train_perfect_predictions))


# Precision and Recall
print(precision_score(y_train_5, y_train_pred)) # == 3530 / (687 + 3530)
print(recall_score(y_train_5, y_train_pred)) # == 3530 / (1891 + 3530)
print(f1_score(y_train_5, y_train_pred))