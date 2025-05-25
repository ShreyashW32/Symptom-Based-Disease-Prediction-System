import numpy as np


class KNN:
    def __init__(self, k=5, num_classes=20):
        self.k = k
        self.num_classes = num_classes
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)  # Convert to NumPy array for safe indexing

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.num_classes))

        for i in range(n_samples):
            # Compute Euclidean distances
            distances = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]  # Shape: (k,)
            # Get labels of k nearest neighbors
            k_labels = self.y_train[k_indices]  # Now safe due to np.array
            # Compute class probabilities based on neighbor counts
            for c in range(self.num_classes):
                probs[i, c] = np.sum(k_labels == c) / self.k

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
