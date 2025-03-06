import numpy as np


class LDA:

    def __init__(self):
        self.means = {}
        self.priors = {}
        self.pooled_cov = None
        self.inv_cov = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        n_samples, n_features = X.shape

        pooled_cov = np.zeros((n_features, n_features))
        for c in classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples
            diff = X_c - self.means[c]
            pooled_cov += diff.T @ diff

        pooled_cov /= (n_samples - len(classes))
        self.pooled_cov = pooled_cov
        self.inv_cov = np.linalg.inv(pooled_cov)

    def predict_proba(self, Xtest):
        Xtest = np.asarray(Xtest)
        mean0 = self.means[0]
        mean1 = self.means[1]
        prior0 = self.priors[0]
        prior1 = self.priors[1]
        inv_cov = self.inv_cov

        # LD function - d_k(x) = x^T * inv_cov * mean_k - 0.5 * mean_k^T * inv_cov * mean_k + log(prior_k)
        d0 = Xtest.dot(inv_cov.dot(mean0)) - 0.5 * (mean0.T.dot(inv_cov.dot(mean0))) + np.log(prior0)
        d1 = Xtest.dot(inv_cov.dot(mean1)) - 0.5 * (mean1.T.dot(inv_cov.dot(mean1))) + np.log(prior1)

        max_d = np.maximum(d0, d1)
        exp_d0 = np.exp(d0 - max_d)
        exp_d1 = np.exp(d1 - max_d)
        prob = exp_d1 / (exp_d0 + exp_d1)
        return prob

    def predict(self, Xtest):
        proba = self.predict_proba(Xtest)
        return (proba >= 0.5).astype(int)

    def get_params(self):
        return [self.means, self.priors, self.pooled_cov]
