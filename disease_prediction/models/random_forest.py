import numpy as np
from .decision_tree import DecisionTree
import random


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, num_classes=20):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                num_classes=self.num_classes)
            X_sample, y_sample = self.bootstrap_sample(
                np.array(X), np.array(y))
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
