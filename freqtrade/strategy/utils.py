__all__ = ['NumpyEncoder', 'get_daily_vol', 'PurgedKFold']

# Cell
import pandas as pd
import numpy as np
import json

from sklearn.model_selection._split import _BaseKFold


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def get_daily_vol(close, span0=100):
    # daily vol, reindexed to cloes
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :]
    )
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


class PurgedKFold(_BaseKFold):
    """
    Extend KFold to work with labels that span intervals
    The train is is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training examples in between
    """

    def __init__(self, n_splits=3, t1=None, pct_embargo=0.0, random_state=None):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label through Dates must be a pandas series")
        super(PurgedKFold, self).__init__(
            n_splits, shuffle=False, random_state=random_state
        )
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError("X and ThruDateValues must have the same index length")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]
        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg :]))
            yield train_indices, test_indices