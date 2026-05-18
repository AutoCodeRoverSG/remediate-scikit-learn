"""
Tests for sklearn.cluster._feature_agglomeration
"""

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_array_almost_equal


def test_feature_agglomeration():
    n_clusters = 1
    X = np.array([0, 0, 1]).reshape(1, 3)  # (n_samples, n_features)

    agglo_mean = FeatureAgglomeration(n_clusters=n_clusters, pooling_func=np.mean)
    agglo_median = FeatureAgglomeration(n_clusters=n_clusters, pooling_func=np.median)
    agglo_mean.fit(X)
    agglo_median.fit(X)

    assert np.size(np.unique(agglo_mean.labels_)) == n_clusters
    assert np.size(np.unique(agglo_median.labels_)) == n_clusters
    assert np.size(agglo_mean.labels_) == X.shape[1]
    assert np.size(agglo_median.labels_) == X.shape[1]

    # Test transform
    xt_mean = agglo_mean.transform(X)
    xt_median = agglo_median.transform(X)
    assert xt_mean.shape[1] == n_clusters
    assert xt_median.shape[1] == n_clusters
    assert xt_mean == np.array([1 / 3.0])
    assert xt_median == np.array([0.0])

    # Test inverse transform
    x_full_mean = agglo_mean.inverse_transform(xt_mean)
    x_full_median = agglo_median.inverse_transform(xt_median)
    assert np.unique(x_full_mean[0]).size == n_clusters
    assert np.unique(x_full_median[0]).size == n_clusters

    assert_array_almost_equal(agglo_mean.transform(x_full_mean), xt_mean)
    assert_array_almost_equal(agglo_median.transform(x_full_median), xt_median)


def test_feature_agglomeration_feature_names_out():
    """Check `get_feature_names_out` for `FeatureAgglomeration`."""
    X, _ = make_blobs(n_features=6, random_state=0)
    agglo = FeatureAgglomeration(n_clusters=3)
    agglo.fit(X)
    n_clusters = agglo.n_clusters_

    names_out = agglo.get_feature_names_out()
    assert_array_equal(
        [f"featureagglomeration{i}" for i in range(n_clusters)], names_out
    )
