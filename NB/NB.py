import numpy as np


class NB:

    def __init__(self):
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        n_samples, _ = X.shape

        for c in classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples

    def predict_proba(self, Xtest):

        Xtest = np.asarray(Xtest)
        n_samples, n_features = Xtest.shape
        log_probs = np.zeros((n_samples, 2))

        eps = 1e-9  # small constant to avoid division by zero

        # calculate log-probabilities for each class
        for idx, c in enumerate([0, 1]):
            mean = self.means[c]
            var = self.variances[c]
            prior = self.priors[c]
            # constant term: -0.5 * sum(log(2*pi*var))
            const = -0.5 * np.sum(np.log(2 * np.pi * var + eps))
            diff = Xtest - mean
            # log-likelihood - summing over features
            ll = const - 0.5 * np.sum((diff ** 2) / (var + eps), axis=1)
            log_probs[:, idx] = ll + np.log(prior)

        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_log = np.exp(log_probs - max_log)
        probs = exp_log / np.sum(exp_log, axis=1, keepdims=True)
        return probs[:, 1]

    def predict(self, Xtest):
        proba = self.predict_proba(Xtest)
        return (proba >= 0.5).astype(int)

    def get_params(self):
        return [self.means, self.variances, self.priors]
