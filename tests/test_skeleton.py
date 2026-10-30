import numpy as np
import pytest

from decision_tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseDecisionTree()


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_hyperparameters(cls):
    model = cls(max_depth=3, min_samples_split=4)
    assert model.get_params()["max_depth"] == 3
    model.set_params(criterion="entropy")
    assert model.criterion == "entropy"
    with pytest.raises(ValueError):
        model.set_params(foo=1)


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_invalid_params(cls):
    for params in ({"max_depth": 0}, {"min_samples_split": 1}, {"criterion": "foo"}):
        with pytest.raises(ValueError):
            cls(**params).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        cls().predict([[0]])


def test_impurity():
    y = np.array([0, 0, 1, 1])
    assert DecisionTreeClassifier(criterion="gini")._impurity(y) == 0.5
    assert DecisionTreeClassifier(criterion="entropy")._impurity(y) == 1.0
    assert DecisionTreeRegressor(criterion="mse")._impurity(np.array([1.0, 2.0, 3.0])) == pytest.approx(2 / 3)
    assert DecisionTreeRegressor(criterion="mae")._impurity(np.array([1.0, 2.0, 10.0])) == 3.0


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_numeric(cls):
    X = [[1], [2], [3], [4]]
    y = [0, 0, 1, 1]
    model = cls().fit(X, y)
    assert model.root.threshold == 2.5
    assert model.eval(X, y) == 1.0


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_max_depth(cls):
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20) % 3
    model = cls(max_depth=1).fit(X, y)
    assert model.root.left.is_leaf and model.root.right.is_leaf


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classifier_one_hot(criterion):
    # One-hot colors: red, green, blue
    X = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    y = ["a", "a", "b", "b", "b", "b"]
    model = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    assert model.root.threshold == 0.5
    assert model.eval(X, y) == 1.0

    # Unseen color (all 0s) goes with the rest
    assert model.predict([[0, 0, 0]])[0] == "b"


def test_classifier_mixed():
    # size + one-hot color
    rng = np.random.default_rng(0)
    size = rng.uniform(0, 10, 200)
    color = rng.integers(0, 3, 200)
    X = np.column_stack([size, np.eye(3)[color]])
    y = (color == 0) & (size > 5)

    model = DecisionTreeClassifier().fit(X, y)
    assert model.eval(X, y) == 1.0
    assert model.predict([[8, 1, 0, 0], [8, 0, 1, 0], [2, 1, 0, 0]]).tolist() == [True, False, False]


@pytest.mark.parametrize("criterion", ["mse", "mae"])
def test_regressor_mixed(criterion):
    # size + one-hot city
    rng = np.random.default_rng(1)
    size = rng.uniform(0, 10, 200)
    city = rng.integers(0, 3, 200)
    X = np.column_stack([size, np.eye(3)[city]])
    y = np.array([100.0, 200.0, 300.0])[city] + np.where(size > 5, 50.0, 0.0)

    model = DecisionTreeRegressor(criterion=criterion).fit(X, y)
    assert model.eval(X, y) == 1.0
    assert model.predict([[9, 1, 0, 0], [1, 0, 0, 1]]).tolist() == [150.0, 300.0]
