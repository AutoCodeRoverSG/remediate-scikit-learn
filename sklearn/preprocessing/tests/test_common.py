import warnings

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    maxabs_scale,
    minmax_scale,
    power_transform,
    quantile_transform,
    robust_scale,
    scale,
)
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import (
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DIA_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)

iris = load_iris()


def _get_valid_samples_by_column(X, col):
    """Get non NaN samples in column of X"""
    return X[:, [col]][~np.isnan(X[:, col])]


@pytest.mark.parametrize(
    "est, func, support_sparse, strictly_positive, omit_kwargs",
    [
        (MaxAbsScaler(), maxabs_scale, True, False, ["clip"]),
        (MinMaxScaler(), minmax_scale, False, False, ["clip"]),
        (StandardScaler(), scale, False, False, []),
        (StandardScaler(with_mean=False), scale, True, False, []),
        (PowerTransformer("yeo-johnson"), power_transform, False, False, []),
        (PowerTransformer("box-cox"), power_transform, False, True, []),
        (QuantileTransformer(n_quantiles=10), quantile_transform, True, False, []),
        (RobustScaler(), robust_scale, False, False, []),
        (RobustScaler(with_centering=False), robust_scale, True, False, []),
    ],
)
def test_missing_value_handling(
    est, func, support_sparse, strictly_positive, omit_kwargs
):
    # check that the preprocessing method let pass nan
    rng = np.random.default_rng(42)
    X = iris.data.copy()
    n_missing = 50
    X[
        rng.integers(X.shape[0], size=n_missing),
        rng.integers(X.shape[1], size=n_missing),
    ] = np.nan
    if strictly_positive:
        X += np.nanmin(X) + 0.1
    X_train, X_test = train_test_split(X, random_state=1)
    # sanity check
    assert not np.all(np.isnan(X_train), axis=0).any()
    assert np.any(np.isnan(X_train), axis=0).all()
    assert np.any(np.isnan(X_test), axis=0).all()
    X_test[:, 0] = np.nan  # make sure this boundary case is tested

    est = clone(est)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x_trans = est.fit(X_train).transform(X_test)
    # ensure no warnings are raised
    # missing values should still be missing, and only them
    assert_array_equal(np.isnan(x_trans), np.isnan(X_test))

    # check that the function leads to the same results as the class
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x_trans_class = est.transform(X_train)
    kwargs = est.get_params()
    # remove the parameters which should be omitted because they
    # are not defined in the counterpart function of the preprocessing class
    for kwarg in omit_kwargs:
        _ = kwargs.pop(kwarg)
    x_trans_func = func(X_train, **kwargs)
    assert_array_equal(np.isnan(x_trans_func), np.isnan(x_trans_class))
    assert_allclose(
        x_trans_func[~np.isnan(x_trans_func)],
        x_trans_class[~np.isnan(x_trans_class)],
    )

    # check that the inverse transform keep NaN
    x_trans_inv = est.inverse_transform(x_trans)
    assert_array_equal(np.isnan(x_trans_inv), np.isnan(X_test))
    # FIXME: we can introduce equal_nan=True in recent version of numpy.
    # For the moment which just check that non-NaN values are almost equal.
    assert_allclose(x_trans_inv[~np.isnan(x_trans_inv)], X_test[~np.isnan(X_test)])

    for i in range(X.shape[1]):
        # train only on non-NaN
        est.fit(_get_valid_samples_by_column(X_train, i))
        # check transforming with NaN works even when training without NaN
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            x_trans_col = est.transform(X_test[:, [i]])
        assert_allclose(x_trans_col, x_trans[:, [i]])
        # check non-NaN is handled as before - the 1st column is all nan
        if not np.isnan(X_test[:, i]).all():
            x_trans_col_nonan = est.transform(
                _get_valid_samples_by_column(X_test, i)
            )
            assert_array_equal(
                x_trans_col_nonan, x_trans_col[~np.isnan(x_trans_col.squeeze())]
            )

    if support_sparse:
        est_dense = clone(est)
        est_sparse = clone(est)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            x_trans_dense = est_dense.fit(X_train).transform(X_test)
            x_trans_inv_dense = est_dense.inverse_transform(x_trans_dense)

        for sparse_container in (
            BSR_CONTAINERS
            + COO_CONTAINERS
            + CSC_CONTAINERS
            + CSR_CONTAINERS
            + DIA_CONTAINERS
            + DOK_CONTAINERS
            + LIL_CONTAINERS
        ):
            # check that the dense and sparse inputs lead to the same results
            # precompute the matrix to avoid catching side warnings
            x_train_sp = sparse_container(X_train)
            x_test_sp = sparse_container(X_test)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                warnings.simplefilter("error", RuntimeWarning)
                x_trans_sp = est_sparse.fit(x_train_sp).transform(x_test_sp)

            assert_allclose(x_trans_sp.toarray(), x_trans_dense)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                warnings.simplefilter("error", RuntimeWarning)
                x_trans_inv_sp = est_sparse.inverse_transform(x_trans_sp)

            assert_allclose(x_trans_inv_sp.toarray(), x_trans_inv_dense)


@pytest.mark.parametrize(
    "est, func",
    [
        (MaxAbsScaler(), maxabs_scale),
        (MinMaxScaler(), minmax_scale),
        (StandardScaler(), scale),
        (StandardScaler(with_mean=False), scale),
        (PowerTransformer("yeo-johnson"), power_transform),
        (
            PowerTransformer("box-cox"),
            power_transform,
        ),
        (QuantileTransformer(n_quantiles=3), quantile_transform),
        (RobustScaler(), robust_scale),
        (RobustScaler(with_centering=False), robust_scale),
    ],
)
def test_missing_value_pandas_na_support(est, func):
    # Test pandas IntegerArray with pd.NA
    pd = pytest.importorskip("pandas")

    X = np.array(
        [
            [1, 2, 3, np.nan, np.nan, 4, 5, 1],
            [np.nan, np.nan, 8, 4, 6, np.nan, np.nan, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    ).T

    # Creates dataframe with IntegerArrays with pd.NA
    x_df = pd.DataFrame(X, dtype="Int16", columns=["a", "b", "c"])
    x_df["c"] = x_df["c"].astype("int")

    x_trans = est.fit_transform(X)
    x_df_trans = est.fit_transform(x_df)

    assert_allclose(x_trans, x_df_trans)
