import pytest

from sklearn.utils._mask import safe_mask
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import check_random_state


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_safe_mask(csr_container):
    random_state = check_random_state(0)
    X = random_state.rand(5, 4)
    x_csr = csr_container(X)
    mask = [False, False, True, True, True]

    mask = safe_mask(X, mask)
    assert X[mask].shape[0] == 3

    mask = safe_mask(x_csr, mask)
    assert x_csr[mask].shape[0] == 3
