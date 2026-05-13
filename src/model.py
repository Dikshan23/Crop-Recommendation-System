import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None, class_distribution=None):
        self.feature            = feature
        self.threshold          = threshold
        self.left               = left
        self.right              = right
        self.value              = value
        self.class_distribution = class_distribution or {}


class DecisionTreeCART:
    def __init__(self, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.random_state       = random_state
        self.root               = None
        self._rng = np.random.RandomState(random_state)

    # ── Gini Impurity ─────────────────────────────────────────────────────────
    def gini(self, y):
        classes  = np.unique(y)
        impurity = 1.0
        for c in classes:
            p         = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    # ── Split data on a feature / threshold ───────────────────────────────────
    def split(self, X, y, feature, threshold):
        left_idx  = X[:, feature] <= threshold
        right_idx = X[:, feature] >  threshold
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    # ── Find best split — with threshold sampling for speed ───────────────────
    def best_split(self, X, y):
        best_feature   = None
        best_threshold = None
        best_gini      = float("inf")
        n_features     = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            # Sample 50 candidate thresholds when there are too many unique values.
            # Continuous features like rainfall / K can have 10,000+ unique values
            # per node — trying all of them makes training take 30+ minutes.
            # Sampling 50 gives a very close approximation in a fraction of the time.
            if len(thresholds) > 50:
                thresholds = self._rng.choice(thresholds, size=50, replace=False)

            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(
                    X, y, feature, threshold
                )

                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                if (len(y_left)  < self.min_samples_leaf or
                        len(y_right) < self.min_samples_leaf):
                    continue

                gini_left     = self.gini(y_left)
                gini_right    = self.gini(y_right)
                weighted_gini = (
                    len(y_left)  / len(y) * gini_left +
                    len(y_right) / len(y) * gini_right
                )

                if weighted_gini < best_gini:
                    best_gini      = weighted_gini
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # ── Leaf helpers ──────────────────────────────────────────────────────────
    def majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def class_distribution(self, y):
        values, counts = np.unique(y, return_counts=True)
        return {str(v): int(c) for v, c in zip(values, counts)}

    # ── Recursive tree builder ────────────────────────────────────────────────
    def build_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_classes = len(np.unique(y))

        # Stopping conditions
        if (depth >= self.max_depth or
                num_classes == 1 or
                num_samples < self.min_samples_split):
            return Node(
                value              = self.majority_class(y),
                class_distribution = self.class_distribution(y)
            )

        feature, threshold = self.best_split(X, y)

        # No valid split found — make a leaf
        if feature is None:
            return Node(
                value              = self.majority_class(y),
                class_distribution = self.class_distribution(y)
            )

        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)

        left_child  = self.build_tree(X_left,  y_left,  depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)

        return Node(feature, threshold, left_child, right_child)

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, node, sample):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self.predict_sample(node.left,  sample)
        else:
            return self.predict_sample(node.right, sample)

    def predict_proba_sample(self, node, sample):
        if node.value is not None:
            dist  = node.class_distribution
            total = sum(dist.values())
            if total == 0:
                return {node.value: 1.0}
            return {cls: count / total for cls, count in dist.items()}
        if sample[node.feature] <= node.threshold:
            return self.predict_proba_sample(node.left,  sample)
        else:
            return self.predict_proba_sample(node.right, sample)

    def predict(self, X):
        return np.array([
            self.predict_sample(self.root, sample) for sample in X
        ])

    def predict_proba(self, X):
        return [
            self.predict_proba_sample(self.root, sample) for sample in X
        ]