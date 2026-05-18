"""Test truncated SVD transformer."""

import numpy as np
import pytest

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
from sklearn.utils.fixes import _sparse_random_array

SVD_SOLVERS = ["arpack", "randomized"]


@pytest.fixture(scope="module")
def x_sparse():
    # Make an X that looks somewhat like a small tf-idf matrix.
    rng = check_random_state(42)
    x = _sparse_random_array((60, 55), density=0.2, format="csr", rng=rng)
    x.data[:] = 1 + np.log(x.data)
    return x


@pytest.mark.parametrize("solver", ["randomized"])
@pytest.mark.parametrize("kind", ("dense", "sparse"))
def test_solvers(x_sparse, solver, kind):
    x = x_sparse if kind == "sparse" else x_sparse.toarray()
    svd_a = TruncatedSVD(30, algorithm="arpack")
    svd = TruncatedSVD(30, algorithm=solver, random_state=42, n_oversamples=100)

    xa = svd_a.fit_transform(x)[:, :6]
    xr = svd.fit_transform(x)[:, :6]
    assert_allclose(xa, xr, rtol=2e-3)

    comp_a = np.abs(svd_a.components_)
    comp = np.abs(svd.components_)
    # All elements are equal, but some elements are more equal than others.
    assert_allclose(comp_a[:9], comp[:9], rtol=1e-3)
    assert_allclose(comp_a[9:], comp[9:], atol=1e-2)


@pytest.mark.parametrize("n_components", (10, 25, 41, 55))
def test_attributes(n_components, x_sparse):
    n_features = x_sparse.shape[1]
    tsvd = TruncatedSVD(n_components).fit(x_sparse)
    assert tsvd.n_components == n_components
    assert tsvd.components_.shape == (n_components, n_features)


@pytest.mark.parametrize(
    "algorithm, n_components",
    [
        ("arpack", 55),
        ("arpack", 56),
        ("randomized", 56),
    ],
)
def test_too_many_components(x_sparse, algorithm, n_components):
    tsvd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
    with pytest.raises(ValueError):
        tsvd.fit(x_sparse)


@pytest.mark.parametrize("fmt", ("array", "csr", "csc", "coo", "lil"))
def test_sparse_formats(fmt, x_sparse):
    n_samples = x_sparse.shape[0]
    x_fmt = x_sparse.toarray() if fmt == "dense" else getattr(x_sparse, "to" + fmt)()
    tsvd = TruncatedSVD(n_components=11)
    x_trans = tsvd.fit_transform(x_fmt)
    assert x_trans.shape == (n_samples, 11)
    x_trans = tsvd.transform(x_fmt)
    assert x_trans.shape == (n_samples, 11)


@pytest.mark.parametrize("algo", SVD_SOLVERS)
def test_inverse_transform(algo, x_sparse):
    # We need a lot of components for the reconstruction to be "almost
    # equal" in all positions. XXX Test means or sums instead?
    tsvd = TruncatedSVD(n_components=52, random_state=42, algorithm=algo)
    x_transformed = tsvd.fit_transform(x_sparse)
    x_inv = tsvd.inverse_transform(x_transformed)
    assert_allclose(x_inv, x_sparse.toarray(), rtol=1e-1, atol=2e-1)


def test_integers(x_sparse):
    n_samples = x_sparse.shape[0]
    x_int = x_sparse.astype(np.int64)
    tsvd = TruncatedSVD(n_components=6)
    x_trans = tsvd.fit_transform(x_int)
    assert x_trans.shape == (n_samples, tsvd.n_components)


@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize("n_components", [10, 20])
@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_explained_variance(x_sparse, kind, n_components, solver):
    x = x_sparse if kind == "sparse" else x_sparse.toarray()
    svd = TruncatedSVD(n_components, algorithm=solver)
    x_tr = svd.fit_transform(x)
    # Assert that all the values are greater than 0
    assert_array_less(0.0, svd.explained_variance_ratio_)

    # Assert that total explained variance is less than 1
    assert_array_less(svd.explained_variance_ratio_.sum(), 1.0)

    # Test that explained_variance is correct
    total_variance = np.var(x_sparse.toarray(), axis=0).sum()
    variances = np.var(x_tr, axis=0)
    true_explained_variance_ratio = variances / total_variance

    assert_allclose(
        svd.explained_variance_ratio_,
        true_explained_variance_ratio,
    )


@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_explained_variance_components_10_20(x_sparse, kind, solver):
    x = x_sparse if kind == "sparse" else x_sparse.toarray()
    svd_10 = TruncatedSVD(10, algorithm=solver, n_iter=10).fit(x)
    svd_20 = TruncatedSVD(20, algorithm=solver, n_iter=10).fit(x)

    # Assert the 1st component is equal
    assert_allclose(
        svd_10.explained_variance_ratio_,
        svd_20.explained_variance_ratio_[:10],
        rtol=5e-3,
    )

    # Assert that 20 components has higher explained variance than 10
    assert (
        svd_20.explained_variance_ratio_.sum() > svd_10.explained_variance_ratio_.sum()
    )


@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_singular_values_consistency(solver, global_random_seed):
    # Check that the TruncatedSVD output has the correct singular values
    rng = np.random.default_rng(global_random_seed)
    n_samples, n_features = 100, 80
    X = rng.standard_normal((n_samples, n_features))

    pca = TruncatedSVD(
        n_components=2, algorithm=solver, random_state=global_random_seed
    ).fit(X)

    # Compare to the Frobenius norm
    x_pca = pca.transform(X)
    assert_allclose(
        np.sum(pca.singular_values_**2.0),
        np.linalg.norm(x_pca, "fro") ** 2.0,
        rtol=1e-2,
    )

    # Compare to the 2-norms of the score vectors
    assert_allclose(
        pca.singular_values_, np.sqrt(np.sum(x_pca**2.0, axis=0)), rtol=1e-2
    )


@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_singular_values_expected(solver, global_random_seed):
    # Set the singular values and see what we get back
    rng = np.random.default_rng(global_random_seed)
    n_samples = 100
    n_features = 110

    X = rng.standard_normal((n_samples, n_features))

    pca = TruncatedSVD(n_components=3, algorithm=solver, random_state=global_random_seed)
    x_pca = pca.fit_transform(X)

    x_pca /= np.sqrt(np.sum(x_pca**2.0, axis=0))
    x_pca[:, 0] *= 3.142
    x_pca[:, 1] *= 2.718

    x_hat_pca = np.dot(x_pca, pca.components_)
    pca.fit(x_hat_pca)
    assert_allclose(pca.singular_values_, [3.142, 2.718, 1.0], rtol=1e-14)


def test_truncated_svd_eq_pca(x_sparse):
    # TruncatedSVD should be equal to PCA on centered data

    x_dense = x_sparse.toarray()

    x_c = x_dense - x_dense.mean(axis=0)

    params = {"n_components": 10, "random_state": 42}

    svd = TruncatedSVD(algorithm="arpack", **params)
    pca = PCA(svd_solver="arpack", **params)

    xt_svd = svd.fit_transform(x_c)
    xt_pca = pca.fit_transform(x_c)

    assert_allclose(xt_svd, xt_pca, rtol=1e-9)
    assert_allclose(pca.mean_, 0, atol=1e-9)
    assert_allclose(svd.components_, pca.components_)


@pytest.mark.parametrize(
    "algorithm, tol, normalizer",
    [
        ("randomized", 0.0, "auto"),
        ("randomized", 0.0, "QR"),
        ("randomized", 0.0, "LU"),
        ("randomized", 0.0, "none"),
        ("arpack", 1e-6, "auto"),
        ("arpack", 0.0, "auto"),
    ],
)
@pytest.mark.parametrize("kind", ("dense", "sparse"))
def test_fit_transform(x_sparse, algorithm, tol, kind, normalizer):
    # fit_transform(X) should equal fit(X).transform(X)
    x = x_sparse if kind == "sparse" else x_sparse.toarray()
    svd = TruncatedSVD(
        n_components=5,
        n_iter=7,
        random_state=42,
        algorithm=algorithm,
        power_iteration_normalizer=normalizer,
        tol=tol,
    )
    x_transformed_1 = svd.fit_transform(x)
    x_transformed_2 = svd.fit(x).transform(x)
    assert_allclose(x_transformed_1, x_transformed_2)
