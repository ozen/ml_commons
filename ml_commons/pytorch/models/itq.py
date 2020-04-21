import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ortho_group


class ITQ:
    def __init__(self, num_bits, num_iterations=100):
        self.num_bits = num_bits
        self.num_iterations = num_iterations
        self.pca = PCA(n_components=self.num_bits)
        self.R = None

    def fit(self, X):
        V = self.pca.fit_transform(X)
        num_samples = V.shape[0]
        self.R = ortho_group.rvs(self.num_bits)
        for i in range(self.num_iterations):
            B = np.ones((num_samples, self.num_bits))
            B[np.dot(V, self.R) < 0] = -1
            S, _, S_hat_T = np.linalg.svd(np.dot(B.T, V))
            self.R = np.dot(S, S_hat_T).T

    def quantize(self, X):
        V = self.pca.transform(X)
        B = np.ones(V.shape)
        B[np.dot(V, self.R) < 0] = -1
        return B
