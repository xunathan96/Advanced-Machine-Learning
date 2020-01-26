import numpy as np
from sklearn.datasets import load_svmlight_file
n_features = 124

def loadData():
    X_train, y_train = load_svmlight_file('a9a', n_features)
    X_test, y_test = load_svmlight_file('a9a.t', n_features)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    return X_train, y_train, X_test, y_test

# LOAD DATA & augment with 1 for bias
X_train, y_train, X_test, y_test = loadData()
X_train[:,n_features-1] = 1
X_test[:,n_features-1] = 1
# change y={-1,+1} to y={0,1}
y_train[y_train==-1] = 0
y_test[y_test==-1] = 0

def sigmoid(X, w):
    mu = 1./(1. + np.exp(- X @ w))
    return mu

def grad_Loss(X, w, y, lamb):
    mu = sigmoid(X, w)
    grad = X.T @ (y - mu) - lamb * w
    return grad

def diagonal_R(mu):
    R_array = np.multiply(mu, 1-mu)
    R = np.diag(R_array)
    return R

def hessian(X, w, lamb):
    mu = sigmoid(X, w)
    R = diagonal_R(mu)
    H = - X.T @ R @ X - lamb*np.identity(n_features)
    return H

def loss(X, w, y, lamb):
    log_likelihood = np.sum(np.multiply(X @ w, y) - np.logaddexp(0, X @ w))
    reg = -0.5*lamb*(w.T @ w)
    L = -(log_likelihood+reg)/y.shape[0]
    return L

def accuracy(X, w, y):
    p_1 = sigmoid(X, w)
    p_1[p_1>=0.5] = 1
    p_1[p_1<0.5] = 0
    correct_predictions = p_1 == y
    return np.sum(correct_predictions)/correct_predictions.shape[0]
