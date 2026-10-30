"""Base class for decision trees with a scikit-learn like API."""
from abc import ABC, abstractmethod

import numpy as np

from .node import Node


class BaseDecisionTree(ABC):
    """Skeleton shared by every tree variant (classification, regression, ...).

    Hyperparameters are set in ``__init__``; ``fit``, ``predict`` and ``eval``
    follow the scikit-learn conventions. Subclasses implement the abstract
    hooks; the TODOs are left to the students.

    Parameters:
        max_depth (int or None): Maximum depth of the tree. The root has depth
            0. ``None`` means nodes grow until they are pure or smaller than
            ``min_samples_split``.
        min_samples_split (int): Minimum number of samples a node needs to be
            considered for splitting. Must be >= 2.
        criterion (str): Name of the impurity measure used to choose splits
            (e.g. 'gini', 'entropy', 'mse'). Valid values depend on the
            subclass.

    Attributes:
        root (Node or None): Root of the fitted tree; ``None`` before ``fit``.
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def get_params(self):
        """Return the hyperparameters.

        Returns:
            dict: Maps 'max_depth', 'min_samples_split' and 'criterion' to
            their current values. Add new hyperparameters here.
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "criterion": self.criterion,
        }

    def set_params(self, **params):
        """Update hyperparameters by keyword, e.g. ``set_params(max_depth=5)``.

        Args:
            **params: Hyperparameter names and new values.

        Returns:
            BaseDecisionTree: ``self``, to allow chaining.

        Raises:
            ValueError: If a name is not a known hyperparameter.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """Build the tree from training data.

        TODO (students): implement this method. It should
          1. validate the hyperparameters (e.g. ``max_depth`` is None or a
             positive int, ``min_samples_split >= 2``, ``criterion`` is
             supported) and raise ``ValueError`` otherwise;
          2. convert ``X`` and ``y`` to arrays and check that they have the
             same number of samples;
          3. call ``self._build_tree(X, y, depth=0)`` and store the returned
             node in ``self.root``;
          4. return ``self``.

        Args:
            X (array-like, shape (n_samples, n_features)): Training features.
            y (array-like, shape (n_samples,)): Targets (class labels for
                classification, real values for regression).

        Returns:
            BaseDecisionTree: ``self`` (the fitted estimator).
        """
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 1):
            raise ValueError("max_depth must be None or a positive int")
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an int >= 2")
        if self.criterion not in self.criteria:
            raise ValueError(f"criterion must be one of {self.criteria}")

        X = np.array(X, dtype=float)
        y = np.array(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        # One-hot (categorical) columns only have 0s and 1s
        self.is_one_hot = [np.isin(X[:, j], [0, 1]).all() for j in range(X.shape[1])]

        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """Predict a target for each row of ``X``.

        TODO (students): implement this method. It should raise an error if
        the model has not been fitted (``self.root is None``) and, for each
        sample, walk from ``self.root``: go left when
        ``x[feature_index] <= threshold`` else right, until a leaf is
        reached, then return that leaf's ``value``.

        Args:
            X (array-like, shape (n_samples, n_features)): Samples to predict.

        Returns:
            array, shape (n_samples,): Predicted target for every sample.
        """
        if self.root is None:
            raise ValueError("Model is not fitted, call fit first")

        X = np.array(X, dtype=float)
        y_pred = []
        for x in X:
            node = self.root
            while not node.is_leaf:
                if x[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(node.value)
        return np.array(y_pred)

    def eval(self, X, y):
        """Score the model on ``(X, y)``.

        TODO (students): implement this method. It should call
        ``self.predict(X)`` and return ``self._score(y, y_pred)``.

        Args:
            X (array-like, shape (n_samples, n_features)): Samples to evaluate.
            y (array-like, shape (n_samples,)): True targets for ``X``.

        Returns:
            float: Metric value (accuracy for classification, R^2 for
            regression; higher is better).
        """
        y_pred = self.predict(X)
        return self._score(np.array(y), y_pred)

    @abstractmethod
    def _build_tree(self, X, y, depth):
        """Recursively grow the tree and return its root ``Node``.

        TODO (students): create a leaf (``Node(value=self._leaf_value(y))``)
        when a stopping condition holds: ``depth`` reached ``max_depth``,
        fewer than ``min_samples_split`` samples, ``y`` is pure, or
        ``_best_split`` finds no useful split. Otherwise split the data on the
        returned (feature, threshold) and recurse on both parts with
        ``depth + 1``.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Depth of this node (root is 0).

        Returns:
            Node: Subtree built from this data.
        """
        # Stopping conditions -> leaf
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._leaf_value(y))
        if len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=self._leaf_value(y))

        split = self._best_split(X, y)
        if split is None:
            return Node(value=self._leaf_value(y))

        feature, threshold = split
        left = X[:, feature] <= threshold
        node = Node(feature_index=feature, threshold=threshold)
        node.left = self._build_tree(X[left], y[left], depth + 1)
        node.right = self._build_tree(X[~left], y[~left], depth + 1)
        return node

    @abstractmethod
    def _impurity(self, y):
        """Compute the impurity of a set of targets.

        TODO (students): implement the measure selected by ``self.criterion``
        (e.g. Gini or entropy for classification, MSE or MAE for regression).
        Lower means more homogeneous; it is used to score candidate splits.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity of ``y`` (0 for a perfectly pure node).
        """

    @abstractmethod
    def _best_split(self, X, y):
        """Find the split that most reduces impurity.

        TODO (students): for each feature and candidate threshold (e.g.
        midpoints between sorted unique values), partition ``y`` into left
        (``x <= threshold``) and right, compute the weighted impurity
        ``(n_l * imp_l + n_r * imp_r) / n`` and keep the split with the
        largest gain over ``self._impurity(y)``. Skip splits leaving a side
        empty.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)`` of the
            best split, or ``None`` if no split improves impurity.
        """
        parent_impurity = self._impurity(y)
        n = len(y)
        best_gain = 0
        best_split = None

        for feature in range(X.shape[1]):
            column = X[:, feature]
            if self.is_one_hot[feature]:
                # One-hot column: category (1) vs rest (0)
                # unseen categories are all 0s so they go with the rest
                thresholds = [0.5]
            else:
                # Midpoints between sorted unique values
                values = np.unique(column)
                thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left = column <= threshold
                n_left = left.sum()
                if n_left == 0 or n_left == n:
                    continue

                # Weighted impurity of the children
                child_impurity = (n_left * self._impurity(y[left]) + (n - n_left) * self._impurity(y[~left])) / n
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)

        return best_split

    @abstractmethod
    def _leaf_value(self, y):
        """Compute the prediction stored in a leaf.

        TODO (students): return the majority class for classification or the
        mean of ``y`` for regression.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The single predicted value for the leaf.
        """

    @abstractmethod
    def _score(self, y_true, y_pred):
        """Compute the metric reported by ``eval``.

        TODO (students): e.g. accuracy for classification, R^2 for regression.

        Args:
            y_true (ndarray, shape (n_samples,)): Ground-truth targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
