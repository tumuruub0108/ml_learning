from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score, precision_recall_curve, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain

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
# print(precision_score(y_train_5, y_train_pred)) # == 3530 / (687 + 3530)
# print(recall_score(y_train_5, y_train_pred)) # == 3530 / (1891 + 3530)
# print(f1_score(y_train_5, y_train_pred))


# The Precision/Recall Trade-off
y_scores = sgd_clf.decision_function([some_digit])
# print(y_scores)

threshold = 3000
y_some_digit_pred = (y_scores > threshold)
# print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
"""
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
plt.show()

"""

idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]


# to make predictions on the training set

y_train_pred_90 = (y_scores >= threshold_for_90_precision)
# print(precision_score(y_train_5, y_train_pred_90))

# print(precision_score(y_train_5, y_train_pred_90))


recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
# print(recall_at_90_precision)



# The ROC Curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
"""
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
[...] # beautify the figure: add labels, grid, legend, arrow, and text
plt.show()

"""

# print(roc_auc_score(y_train_5, y_scores))




# RandomForestClassifier,
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
# print(y_probas_forest[:2])


y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

"""
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.show()

"""

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5 # positive proba ≥ 50%
# print(roc_auc_score(y_train_5, y_scores_forest))


# Multiclass Classification
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000]) # y_train, not y_train_5
# print(svm_clf.predict([some_digit]))

some_digit_scores = svm_clf.decision_function([some_digit])
# print(some_digit_scores.round(2)) # The highest score is 9.3,

class_id = some_digit_scores.argmax()
# print(class_id)
# print(svm_clf.classes_)
# print(svm_clf.classes_[class_id])

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

# print(ovr_clf.predict([some_digit]))
# print(len(ovr_clf.estimators_))

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([some_digit]))s

# sprint(sgd_clf.decision_function([some_digit]).round())

# print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# Error Analysis

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
"""

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true", values_format=".0%")
plt.show()

"""

sample_weight = (y_train_pred != y_train)
"""

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, sample_weight=sample_weight, normalize="true", values_format=".0%")
plt.show()

"""

# Multilabel Classification
y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
# print(knn_clf.predict([some_digit]))

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
# print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))


chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
# print(chain_clf.predict([some_digit]))


# Multioutput Classification
np.random.seed(42) # to make this code example reproducible
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
"""

plot_digit(clean_digit)
plt.show()

"""