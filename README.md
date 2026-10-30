# DecisionTree-Implementation

Skeleton for a course activity: implement a decision tree from scratch with a
scikit-learn like API (`fit`, `predict`, `eval`).

- `decision_tree/base.py`: `BaseDecisionTree` with hyperparameters
  (`max_depth`, `min_samples_split`, `criterion`) and abstract hooks.
- `decision_tree/classifier.py`, `regressor.py`: subclasses to complete.
- `decision_tree/node.py`: tree node.

Search for `TODO` / `NotImplementedError` to find what to implement.
Run `pytest` to check the skeleton; extend the tests as you implement.

## Suggested Implementation Roadmap

Build the tree incrementally; finish and test each phase before moving on.

- [x] **Phase 1: Minimal classification tree**
  - [x] Implement one impurity criterion (e.g. Gini).
  - [x] Implement a basic split search (try every candidate split, keep the best).
  - [x] Grow the tree recursively, stop on pure nodes, and store the majority class in leaves.
  - [x] Implement `predict` by traversing the tree.
- [x] **Phase 2: Numeric and categorical features**
  - [x] Numeric: binary splits on a threshold (`x <= t`), using midpoints between sorted unique values.
  - [x] Categorical: one-vs-rest splits (`x == c`) or subset/grouped splits.
  - [x] Detect feature types (or let the user declare them) and route each to the right split search.
  - [x] Decide how to handle categories unseen during `fit` at inference time (e.g. send to the majority child or return the node's prediction).
- [x] **Phase 3: Second classification criterion**
  - [x] Support both Gini and Entropy via the `criterion` hyperparameter.
  - [x] Validate the criterion name and compare the resulting trees.
- [ ] **Phase 4: Hyperparameters**
  - [ ] `max_depth`, `min_samples_split`, `min_samples_leaf`.
  - [ ] `max_features` (random feature subset per split) and, optionally, `min_impurity_decrease`.
  - [x] Validate parameter values and test edge cases (e.g. `max_depth=0`).
- [ ] **Phase 5: Validation, evaluation and tests**
  - [x] Implement `eval` with metrics such as accuracy.
  - [ ] Add a train/validation split or cross-validation helper.
  - [x] Write unit tests: impurity values, split search, numeric/categorical data, unseen categories, hyperparameter limits.
- [x] **Phase 6: Regression tree**
  - [x] Use variance / MSE reduction as the split criterion (optionally MAE).
  - [x] Store the mean (or median) of the targets in leaves.
  - [x] Reuse the numeric and categorical split logic; evaluate with MSE / R².
- [ ] **Phase 7: Optional advanced enhancements**
  - [ ] Pruning (pre- or post-pruning, e.g. cost-complexity).
  - [ ] Feature importance.
  - [ ] Missing-value strategy (e.g. imputation or sending to the larger child).
  - [ ] Serialization (save/load the tree, e.g. to JSON).
