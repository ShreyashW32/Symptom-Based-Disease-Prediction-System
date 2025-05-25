import numpy as np


class MultinomialNB:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.class_probs = None
        self.word_probs = None
        self.classes = None
        self.V = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.V = n_features

        # Initialize class probabilities and word probabilities
        self.class_probs = np.zeros(n_classes)
        self.word_probs = np.zeros((n_classes, n_features))

        # Compute class probabilities and word probabilities
        for c in range(n_classes):
            class_indices = np.where(y == self.classes[c])[0]
            self.class_probs[c] = len(class_indices) / n_samples
            # Sum features for class c (shape: (n_features,))
            class_word_counts = np.sum(X[class_indices], axis=0)
            # Apply Laplace smoothing
            self.word_probs[c] = (class_word_counts + self.alpha) / \
                (np.sum(class_word_counts) + self.alpha * self.V)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, len(self.classes)))

        for i in range(n_samples):
            for c in range(len(self.classes)):
                # Log probability to avoid underflow
                log_prob = np.log(self.class_probs[c])
                log_prob += np.sum(X[i] * np.log(self.word_probs[c]))
                probs[i, c] = log_prob
            # Normalize probabilities
            probs[i] = np.exp(probs[i] - np.log(np.sum(np.exp(probs[i]))))
        return probs

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
