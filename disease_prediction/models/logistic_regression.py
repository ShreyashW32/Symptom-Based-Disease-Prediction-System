import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, num_classes=2):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_classes = num_classes
        self.weights = None
        self.bias = None

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, self.num_classes))
        self.bias = np.zeros(self.num_classes)
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(linear_model)
            y_onehot = np.zeros((n_samples, self.num_classes))
            y_onehot[np.arange(n_samples), y] = 1
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_onehot))
            db = (1 / n_samples) * np.sum(y_pred - y_onehot, axis=0)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.softmax(linear_model)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
