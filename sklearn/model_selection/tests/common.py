"""
Common utilities for testing model selection.
"""

import numpy as np

from sklearn.model_selection import KFold


class OneTimeSplitter:
    """A wrapper to make KFold single entry cv iterator"""

    def __init__(self, n_splits=4, n_samples=99):
        self.n_splits = n_splits
        self.n_samples = n_samples
        self.indices = iter(KFold(n_splits=n_splits).split(np.ones(n_samples)))

    def split(self, X=None, y=None, groups=None):  # noqa
        """Split can be called only once
        X, y, and groups are unused but kept for API compatibility."""
        for index in self.indices:
            yield index

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa
        """Get the number of splits. X, y, groups are part of the splitter API."""
        return self.n_splits
