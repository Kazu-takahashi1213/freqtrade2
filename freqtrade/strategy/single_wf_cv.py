__all__ = ['SinglePurgedWalkForwardCV']

# Cell
import numbers


# We're subclassing the PurgedWalkForwardCV from timeseriescv to allow us to re-use the logic for singular splits
from timeseriescv.cross_validation import PurgedWalkForwardCV

class SinglePurgedWalkForwardCV(PurgedWalkForwardCV):
    def __init__(self, n_splits=10, n_test_splits=1, min_train_splits=2, max_train_splits=None):
        super().__init__(n_splits)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_splits} of type "
                             f"{type(n_test_splits)} was passed.")
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits >= self.n_splits - 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"n_test_splits between 1 and n_splits - 1, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits

        if not isinstance(min_train_splits, numbers.Integral):
            raise ValueError(f"The minimal number of train folds must be of Integral type. {min_train_splits} of type "
                             f"{type(min_train_splits)} was passed.")
        min_train_splits = int(min_train_splits)
        if min_train_splits <= 0 or min_train_splits > self.n_splits - self.n_test_splits:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"min_train_splits between 1 and n_splits - n_test_splits, got min_train_splits = "
                             f"{min_train_splits}.")
        self.min_train_splits = min_train_splits

        if max_train_splits is None:
            max_train_splits = self.n_splits - self.n_test_splits
        if not isinstance(max_train_splits, numbers.Integral):
            raise ValueError(f"The maximal number of train folds must be of Integral type. {max_train_splits} of type "
                             f"{type(max_train_splits)} was passed.")
        max_train_splits = int(max_train_splits)
        if max_train_splits <= 0 or max_train_splits > self.n_splits - self.n_test_splits:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"max_train_split between 1 and n_splits - n_test_splits, got max_train_split = "
                             f"{max_train_splits}.")
        self.max_train_splits = max_train_splits
        self.fold_bounds = []