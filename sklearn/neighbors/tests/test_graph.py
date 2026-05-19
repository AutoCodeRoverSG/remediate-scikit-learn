import numpy as np
import pytest

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.neighbors._base import _is_sorted_by_data
from sklearn.utils._testing import assert_array_equal


def test_transformer_result():
    # Test the number of neighbors returned
    n_neighbors = 5
    n_samples_fit = 20
    n_queries = 18
    n_features = 10

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples_fit, n_features))
    X2 = rng.standard_normal((n_queries, n_features))
    radius = np.percentile(euclidean_distances(X), 10)

    # with n_neighbors
    for mode in ["distance", "connectivity"]:
        add_one = mode == "distance"
        nnt = KNeighborsTransformer(n_neighbors=n_neighbors, mode=mode)
        xt = nnt.fit_transform(X)
        assert xt.shape == (n_samples_fit, n_samples_fit)
        assert xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)
        assert xt.format == "csr"
        assert _is_sorted_by_data(xt)

        x2t = nnt.transform(X2)
        assert x2t.shape == (n_queries, n_samples_fit)
        assert x2t.data.shape == (n_queries * (n_neighbors + add_one),)
        assert x2t.format == "csr"
        assert _is_sorted_by_data(x2t)

    # with radius
    for mode in ["distance", "connectivity"]:
        add_one = mode == "distance"
        nnt = RadiusNeighborsTransformer(radius=radius, mode=mode)
        xt = nnt.fit_transform(X)
        assert xt.shape == (n_samples_fit, n_samples_fit)
        assert xt.data.shape != (n_samples_fit * (n_neighbors + add_one),)
        assert xt.format == "csr"
        assert _is_sorted_by_data(xt)

        x2t = nnt.transform(X2)
        assert x2t.shape == (n_queries, n_samples_fit)
        assert x2t.data.shape != (n_queries * (n_neighbors + add_one),)
        assert x2t.format == "csr"
        assert _is_sorted_by_data(x2t)


def _has_explicit_diagonal(X):
    """Return True if the diagonal is explicitly stored"""
    X = X.tocoo()
    explicit = X.row[X.row == X.col]
    return len(explicit) == X.shape[0]


def test_explicit_diagonal():
    # Test that the diagonal is explicitly stored in the sparse graph
    n_neighbors = 5
    n_samples_fit, n_samples_transform, n_features = 20, 18, 10
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples_fit, n_features))
    X2 = rng.standard_normal((n_samples_transform, n_features))

    nnt = KNeighborsTransformer(n_neighbors=n_neighbors)
    x_trans = nnt.fit_transform(X)
    assert _has_explicit_diagonal(x_trans)
    assert np.all(x_trans.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)

    x_trans = nnt.transform(X)
    assert _has_explicit_diagonal(x_trans)
    assert np.all(x_trans.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)

    # Using transform on new data should not always have zero diagonal
    x2_trans = nnt.transform(X2)
    assert not _has_explicit_diagonal(x2_trans)


@pytest.mark.parametrize("klass", [KNeighborsTransformer, RadiusNeighborsTransformer])
def test_graph_feature_names_out(klass):
    """Check `get_feature_names_out` for transformers defined in `_graph.py`."""

    n_samples_fit = 20
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples_fit, n_features))

    est = klass().fit(X)
    names_out = est.get_feature_names_out()

    class_name_lower = klass.__name__.lower()
    expected_names_out = np.array(
        [f"{class_name_lower}{i}" for i in range(est.n_samples_fit_)],
        dtype=object,
    )
    assert_array_equal(names_out, expected_names_out)
