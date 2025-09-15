import numpy as np
from sklearn.preprocessing import add_dummy_feature


np.random.seed(42) # to make this code example reproducible
m = 100 # number of instances
X = 2 * np.random.rand(m, 1) # column vector
y = 4 + 3 * X + np.random.randn(m, 1) # column vector

X_b = add_dummy_feature(X) # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(theta_best)

X_new = np.array([[0]])


X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new) 
y_predict = X_new_b @ theta_best
print(y_predict)