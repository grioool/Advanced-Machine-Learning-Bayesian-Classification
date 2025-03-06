import numpy as np
from sklearn.metrics import accuracy_score

from LDA import LDA
from NB import NB
from QDA import QDA


# Scheme 1: n = 1000 observations,
# p = 2 features and a binary variable that is generated from the Bernoulli distribution with probability of success 0.5.
# Features of the observations from the class 0 are generated inde- pendently from a normal standard distribution (mean 0, variance 1).
# Features of the observations from the class 1 are generated independently from a normal distribution (mean a, variance 1).
# LDA assumptions are satisfied.

# Scheme 2: Each dataset contain n = 1000 observations,
# p = 2 features and a binary variable that is generated from the Bernoulli distribution with probability of success 0.5.
# Features of the observations from the class 0 are generated from a two-dimensional normal distribution (mean 0, variance 1, correlation ρ).
# Features of the observations from the class 1 are generated from a two-dimensional normal distribution (mean a, variance 1, correlation −ρ).
# LDA assumptions are not satisfied.


def generate_scheme1(n, a):
    y = np.random.binomial(1, 0.5, n)
    X = np.zeros((n, 2))
    for i in range(n):
        if y[i] == 0:
            X[i, :] = np.random.normal(0, 1, 2)
        else:
            X[i, :] = np.random.normal(a, 1, 2)
    return X, y


def generate_scheme2(n, a, rho):
    cov0 = [[1, rho], [rho, 1]]
    cov1 = [[1, -rho], [-rho, 1]]
    y = np.random.binomial(1, 0.5, n)
    X = np.zeros((n, 2))
    for i in range(n):
        if y[i] == 0:
            X[i, :] = np.random.multivariate_normal([0, 0], cov0)
        else:
            X[i, :] = np.random.multivariate_normal([a, a], cov1)
    return X, y


def evaluate_methods(X_train, y_train, X_test, y_test):
    models = {
        'LDA': LDA.LDA(),
        'QDA': QDA.QDA(),
        'NB': NB.NB()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
    return results
