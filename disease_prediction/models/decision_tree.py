import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None, num_classes=20):
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.tree = None

    def gini(self, y):
        p = np.bincount(y, minlength=self.num_classes) / len(y)
        return 1 - np.sum(p ** 2)

    def split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split(
                    X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gini = (len(y_left) * self.gini(y_left) +
                        len(y_right) * self.gini(y_right)) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth) or len(y) < 2:
            prob = np.zeros(self.num_classes)
            unique, counts = np.unique(y, return_counts=True)
            prob[unique] = counts / len(y)
            return {'value': np.argmax(prob), 'prob': prob}

        feature, threshold = self.best_split(X, y)
        if feature is None:
            prob = np.zeros(self.num_classes)
            unique, counts = np.unique(y, return_counts=True)
            prob[unique] = counts / len(y)
            return {'value': np.argmax(prob), 'prob': prob}

        X_left, y_left, X_right, y_right = self.split(X, y, feature, threshold)
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X_left, y_left, depth + 1),
            'right': self.build_tree(X_right, y_right, depth + 1)
        }

    def fit(self, X, y):
        if self.num_classes is None:
            self.num_classes = len(np.unique(y))
        self.tree = self.build_tree(np.array(X), np.array(y))

    def predict_proba(self, X):
        X = np.array(X)
        probs = []
        for x in X:
            node = self.tree
            while 'value' not in node:
                if x[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            probs.append(node['prob'])
        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
