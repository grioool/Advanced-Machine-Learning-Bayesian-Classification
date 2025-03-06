import numpy as np


class QDA:

    def __init__(self):
        self.means = {}
        self.covariances = {}
        self.priors = {}

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        n_samples, _ = X.shape

        for c in classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples
            self.covariances[c] = np.cov(X_c, rowvar=False, bias=False)

    def predict_proba(self, Xtest):
        Xtest = np.asarray(Xtest)
        n_samples = Xtest.shape[0]
        scores = {0: np.zeros(n_samples), 1: np.zeros(n_samples)}

        # discriminant score using the quadratic form:
        for c in [0, 1]:
            mean = self.means[c]
            cov = self.covariances[c]
            prior = self.priors[c]
            inv_cov = np.linalg.inv(cov)
            log_det = np.log(np.linalg.det(cov))
            diff = Xtest - mean
            # quadratic term: (x - mean)^T * inv_cov * (x - mean)
            quad = np.einsum('ij,ij->i', diff.dot(inv_cov), diff)
            scores[c] = -0.5 * (log_det + quad) + np.log(prior)

        max_scores = np.maximum(scores[0], scores[1])
        exp0 = np.exp(scores[0] - max_scores)
        exp1 = np.exp(scores[1] - max_scores)
        prob = exp1 / (exp0 + exp1)
        return prob

    def predict(self, Xtest):
        proba = self.predict_proba(Xtest)
        return (proba >= 0.5).astype(int)

    def get_params(self):
        return [self.means, self.priors, self.covariances]
