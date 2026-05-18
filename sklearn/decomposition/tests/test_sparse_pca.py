# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.datasets import make_low_rank_matrix
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    if_safe_multiprocessing_with_blas,
)
from sklearn.utils.extmath import svd_flip


def generate_toy_data(n_components, n_samples, image_size, random_state=None):
    n_features = image_size[0] * image_size[1]

    rng = check_random_state(random_state)
    U = rng.randn(n_samples, n_components)
    V = rng.randn(n_components, n_features)

    centers = [(3, 3), (6, 7), (8, 1)]
    sz = [1, 2, 1]
    for k in range(n_components):
        img = np.zeros(image_size)
        xmin, xmax = centers[k][0] - sz[k], centers[k][0] + sz[k]
        ymin, ymax = centers[k][1] - sz[k], centers[k][1] + sz[k]
        img[xmin:xmax][:, ymin:ymax] = 1.0
        V[k, :] = img.ravel()

    # Y is defined by : Y = UV + noise
    Y = np.dot(U, V)
    Y += 0.1 * rng.randn(Y.shape[0], Y.shape[1])  # Add noise
    return Y, U, V


# SparsePCA can be a bit slow. To avoid having test times go up, we
# test different aspects of the code in the same test


def test_correct_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 10))
    spca = SparsePCA(n_components=8, random_state=0)
    U = spca.fit_transform(X)
    assert spca.components_.shape == (8, 10)
    assert U.shape == (12, 8)
    # test overcomplete decomposition
    spca = SparsePCA(n_components=13, random_state=0)
    U = spca.fit_transform(X)
    assert spca.components_.shape == (13, 10)
    assert U.shape == (12, 13)


def test_fit_transform(global_random_seed):
    alpha = 1
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=global_random_seed)  # wide array
    spca_lars = SparsePCA(
        n_components=3, method="lars", alpha=alpha, random_state=global_random_seed
    )
    spca_lars.fit(Y)

    # Test that CD gives similar results
    spca_lasso = SparsePCA(
        n_components=3, method="cd", random_state=global_random_seed, alpha=alpha
    )
    spca_lasso.fit(Y)
    assert_allclose(spca_lasso.components_, spca_lars.components_, rtol=5e-4, atol=2e-4)


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
@if_safe_multiprocessing_with_blas
def test_fit_transform_parallel(global_random_seed):
    alpha = 1
    Y, _, _ = generate_toy_data(
        3, 10, (8, 8), random_state=global_random_seed
    )  # wide array
    spca_lars = SparsePCA(
        n_components=3, method="lars", alpha=alpha, random_state=global_random_seed
    )
    spca_lars.fit(Y)
    U1 = spca_lars.transform(Y)
    # Test multiple CPUs
    spca = SparsePCA(
        n_components=3,
        n_jobs=2,
        method="lars",
        alpha=alpha,
        random_state=global_random_seed,
    ).fit(Y)
    U2 = spca.transform(Y)
    assert not np.all(spca_lars.components_ == 0)
    assert_array_almost_equal(U1, U2)


def test_transform_nan(global_random_seed):
    # Test that SparsePCA won't return NaN when there is 0 feature in all
    # samples.
    Y, _, _ = generate_toy_data(
        3, 10, (8, 8), random_state=global_random_seed
    )  # wide array
    Y[:, 0] = 0
    estimator = SparsePCA(n_components=8, random_state=global_random_seed)
    assert not np.any(np.isnan(estimator.fit_transform(Y)))


def test_fit_transform_tall(global_random_seed):
    rng = np.random.default_rng(global_random_seed)
    seeds = rng.integers(0, 2**31, size=3)
    Y, _, _ = generate_toy_data(3, 65, (8, 8), random_state=seeds[0])  # tall array
    spca_lars = SparsePCA(n_components=3, method="lars", random_state=seeds[1])
    U1 = spca_lars.fit_transform(Y)
    spca_lasso = SparsePCA(n_components=3, method="cd", random_state=seeds[2])
    U2 = spca_lasso.fit(Y).transform(Y)
    assert_allclose(U1, U2, rtol=1e-4, atol=2e-5)


def test_initialization(global_random_seed):
    rng = np.random.default_rng(global_random_seed)
    u_init = rng.standard_normal((5, 3))
    v_init = rng.standard_normal((3, 4))
    model = SparsePCA(
        n_components=3,
        u_init=u_init,
        v_init=v_init,
        max_iter=0,
        random_state=global_random_seed,
    )
    model.fit(rng.standard_normal((5, 4)))

    expected_components = v_init / np.linalg.norm(v_init, axis=1, keepdims=True)
    expected_components = svd_flip(u=expected_components.T, v=None)[0].T
    assert_allclose(model.components_, expected_components)


def test_mini_batch_correct_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 10))
    pca = MiniBatchSparsePCA(n_components=8, max_iter=1, random_state=0)
    U = pca.fit_transform(X)
    assert pca.components_.shape == (8, 10)
    assert U.shape == (12, 8)
    # test overcomplete decomposition
    pca = MiniBatchSparsePCA(n_components=13, max_iter=1, random_state=1)
    U = pca.fit_transform(X)
    assert pca.components_.shape == (13, 10)
    assert U.shape == (12, 13)


def test_scaling_fit_transform(global_random_seed):
    alpha = 1
    rng = np.random.default_rng(global_random_seed)
    seed1, seed2 = rng.integers(0, 2**31, size=2)
    Y, _, _ = generate_toy_data(3, 1000, (8, 8), random_state=seed1)
    spca_lars = SparsePCA(n_components=3, method="lars", alpha=alpha, random_state=seed2)
    results_train = spca_lars.fit_transform(Y)
    results_test = spca_lars.transform(Y[:10])
    assert_allclose(results_train[0], results_test[0])


def test_pca_vs_spca(global_random_seed):
    rng = np.random.default_rng(global_random_seed)
    seeds = rng.integers(0, 2**31, size=4)
    Y, _, _ = generate_toy_data(3, 1000, (8, 8), random_state=seeds[0])
    Z, _, _ = generate_toy_data(3, 10, (8, 8), random_state=seeds[1])
    spca = SparsePCA(alpha=0, ridge_alpha=0, n_components=2, random_state=seeds[2])
    pca = PCA(n_components=2, random_state=seeds[3])
    pca.fit(Y)
    spca.fit(Y)
    results_test_pca = pca.transform(Z)
    results_test_spca = spca.transform(Z)
    assert_allclose(
        np.abs(spca.components_.dot(pca.components_.T)), np.eye(2), atol=1e-4
    )
    results_test_pca *= np.sign(results_test_pca[0, :])
    results_test_spca *= np.sign(results_test_spca[0, :])
    assert_allclose(results_test_pca, results_test_spca, atol=1e-4)


@pytest.mark.parametrize("spca_cls", [SparsePCA, MiniBatchSparsePCA])
@pytest.mark.parametrize("n_components", [None, 3])
def test_spca_n_components_(spca_cls, n_components):
    rng = np.random.default_rng(0)
    n_samples, n_features = 12, 10
    X = rng.standard_normal((n_samples, n_features))

    model = spca_cls(n_components=n_components).fit(X)

    if n_components is not None:
        assert model.n_components_ == n_components
    else:
        assert model.n_components_ == n_features


@pytest.mark.parametrize("spca", (SparsePCA, MiniBatchSparsePCA))
@pytest.mark.parametrize("method", ("lars", "cd"))
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_sparse_pca_dtype_match(spca, method, data_type, expected_type):
    # Verify output matrix dtype
    n_samples, n_features, n_components = 12, 10, 3
    rng = np.random.default_rng(0)
    input_array = rng.standard_normal((n_samples, n_features)).astype(data_type)
    model = spca(n_components=n_components, method=method)
    transformed = model.fit_transform(input_array)

    assert transformed.dtype == expected_type
    assert model.components_.dtype == expected_type


@pytest.mark.parametrize("spca_estimator", (SparsePCA, MiniBatchSparsePCA))
@pytest.mark.parametrize("method", ("lars", "cd"))
def test_sparse_pca_numerical_consistency(spca_estimator, method, global_random_seed):
    # Verify numericall consistentency among np.float32 and np.float64
    n_samples, n_features, n_components = 20, 20, 5
    input_array = make_low_rank_matrix(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=n_components,
        random_state=global_random_seed,
    )

    model_32 = spca_estimator(
        n_components=n_components,
        method=method,
        random_state=global_random_seed,
    )
    transformed_32 = model_32.fit_transform(input_array.astype(np.float32))

    model_64 = spca_estimator(
        n_components=n_components,
        method=method,
        random_state=global_random_seed,
    )
    transformed_64 = model_64.fit_transform(input_array.astype(np.float64))
    assert_allclose(transformed_64, transformed_32)
    assert_allclose(model_64.components_, model_32.components_)


@pytest.mark.parametrize("spca_model", [SparsePCA, MiniBatchSparsePCA])
def test_spca_feature_names_out(spca_model):
    """Check feature names out for *SparsePCA."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 12, 10
    X = rng.standard_normal((n_samples, n_features))

    model = spca_model(n_components=4).fit(X)
    names = model.get_feature_names_out()

    estimator_name = spca_model.__name__.lower()
    assert_array_equal([f"{estimator_name}{i}" for i in range(4)], names)


def test_spca_early_stopping(global_random_seed):
    """Check that `tol` and `max_no_improvement` act as early stopping."""
    rng = np.random.RandomState(global_random_seed)
    n_samples, n_features = 50, 10
    X = rng.randn(n_samples, n_features)

    # vary the tolerance to force the early stopping of one of the model
    model_early_stopped = MiniBatchSparsePCA(
        max_iter=100, tol=0.5, random_state=global_random_seed
    ).fit(X)
    model_not_early_stopped = MiniBatchSparsePCA(
        max_iter=100, tol=1e-3, random_state=global_random_seed
    ).fit(X)
    assert model_early_stopped.n_iter_ < model_not_early_stopped.n_iter_

    # force the max number of no improvement to a large value to check that
    # it does help to early stop
    model_early_stopped = MiniBatchSparsePCA(
        max_iter=100, tol=1e-6, max_no_improvement=2, random_state=global_random_seed
    ).fit(X)
    model_not_early_stopped = MiniBatchSparsePCA(
        max_iter=100, tol=1e-6, max_no_improvement=100, random_state=global_random_seed
    ).fit(X)
    assert model_early_stopped.n_iter_ < model_not_early_stopped.n_iter_


def test_equivalence_components_pca_spca(global_random_seed):
    """Check the equivalence of the components found by PCA and SparsePCA.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23932
    """
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(50, 4)

    n_components = 2
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=0,
    ).fit(X)
    spca = SparsePCA(
        n_components=n_components,
        method="lars",
        ridge_alpha=0,
        alpha=0,
        random_state=0,
    ).fit(X)

    assert_allclose(pca.components_, spca.components_)


def test_sparse_pca_inverse_transform(global_random_seed):
    """Check that `inverse_transform` in `SparsePCA` and `PCA` are similar."""
    rng = np.random.RandomState(global_random_seed)
    n_samples, n_features = 10, 5
    X = rng.randn(n_samples, n_features)

    n_components = 2
    spca = SparsePCA(
        n_components=n_components,
        alpha=1e-12,
        ridge_alpha=1e-12,
        random_state=global_random_seed,
    )
    pca = PCA(n_components=n_components, random_state=global_random_seed)
    x_trans_spca = spca.fit_transform(X)
    x_trans_pca = pca.fit_transform(X)
    assert_allclose(
        spca.inverse_transform(x_trans_spca), pca.inverse_transform(x_trans_pca)
    )


@pytest.mark.parametrize("spca_estimator", [SparsePCA, MiniBatchSparsePCA])
def test_transform_inverse_transform_round_trip(spca_estimator, global_random_seed):
    """Check the `transform` and `inverse_transform` round trip with no loss of
    information.
    """
    rng = np.random.RandomState(global_random_seed)
    n_samples, n_features = 10, 5
    X = rng.randn(n_samples, n_features)

    n_components = n_features
    spca = spca_estimator(
        n_components=n_components,
        alpha=1e-12,
        ridge_alpha=1e-12,
        random_state=global_random_seed,
    )
    x_trans_spca = spca.fit_transform(X)
    assert_allclose(spca.inverse_transform(x_trans_spca), X)
