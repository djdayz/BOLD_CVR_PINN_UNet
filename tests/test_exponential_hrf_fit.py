import pytest

np = pytest.importorskip("numpy")

from hybrid_cvr.cvr.exponential_hrf import precompute_ode_regressors


def test_exponential_hrf_regressor_shapes():
    time = np.arange(10, dtype=float)
    etco2 = np.ones(10)
    regs = precompute_ode_regressors(etco2, time, np.array([0, 2]), np.array([5, 10, 20]))
    assert regs.shape == (2, 3, 10)
