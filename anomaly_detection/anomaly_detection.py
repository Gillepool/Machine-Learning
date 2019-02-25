import numpy as np
import matplotlib.pyplot as plt

def multivariate_gaussian(X, mu, sigma):
    k = len(mu)
    if np.ndim(sigma) == 1:
        sigma = np.diag(sigma)

    X = X - mu.reshape(mu.size, order='F').T

    p = np.dot(np.power(2 * np.pi, - k / 2.0), np.power(np.linalg.det(sigma), -0.5) ) * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(sigma)) * X, axis=1))
    return p

def estimate_gaussian(X):
    _, n = X.shape
    mu = np.zeros((n, 1))
    sigma = np.zeros((n, 1))

    mu = np.mean(X, axis=0).T
    sigma = np.var(X, axis=0).T

    return mu, sigma

def select_threshold(yVal, pVal):
    best_eps = 0
    best_f1 = 0

    step_size = (max(pVal) - min(pVal)) / 1000

    for epsilon in np.arange(min(pVal), max(pVal), step_size):
        cv_predictions = pVal < epsilon

        true_pos = np.sum(np.logical_and((cv_predictions==1), (yVal==1)).astype(float))
        false_pos = np.sum(np.logical_and((cv_predictions==1), (yVal==0)).astype(float))
        false_neg = np.sum(np.logical_and((cv_predictions==0), (yVal==1)).astype(float))

        precision = true_pos/(true_pos+false_pos)
        recall = true_pos/(true_pos+false_neg)
        F1 = (2*precision*recall)/(precision+recall)

        if F1 > best_f1:
            best_f1 = F1
            best_eps = epsilon

    return best_eps, best_f1
