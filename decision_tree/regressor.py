"""Regression tree skeleton."""
import numpy as np

from .base import BaseDecisionTree


class DecisionTreeRegressor(BaseDecisionTree):
    """Tree variant for regression.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'mse', 'mae'.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """

    criteria = ["mse", "mae"]

    def __init__(self, max_depth=None, min_samples_split=2, criterion="mse"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
        """Recursively grow the tree.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Current depth (root is 0).

        Returns:
            Node: Leaf (when a stopping rule triggers) or internal node with
            two subtrees built from the best split.
        """
        return super()._build_tree(X, y, depth)

    def _impurity(self, y):
        """Impurity of ``y`` per ``self.criterion``: mean squared error or mean absolute error of y around its center (mean for 'mse', median for 'mae').

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        if self.criterion == "mse":
            return np.mean((y - np.mean(y)) ** 2)
        return np.mean(np.abs(y - np.median(y)))

    def _best_split(self, X, y):
        """Search all features/thresholds for the largest impurity decrease.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)``, or
            None if no split reduces impurity.
        """
        return super()._best_split(X, y)

    def _leaf_value(self, y):
        """Return the mean of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """
        return np.mean(y)

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: R^2 = 1 - SS_res / SS_tot.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot
