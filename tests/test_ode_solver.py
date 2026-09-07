import pytest

np = pytest.importorskip("numpy")

from hybrid_cvr.cvr.ode import solve_ode_response_numpy
from hybrid_cvr.pinn.torch_ode import simulate_ode_bold_torch


def test_zero_cvr_gives_near_zero_response():
    time = np.arange(20, dtype=float)
    etco2 = np.ones(20) * 5
    y = solve_ode_response_numpy(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2)) * 10, etco2, time)
    assert np.allclose(y, 0)


def test_larger_cvr_gives_larger_response_and_larger_T_slower():
    time = np.arange(30, dtype=float)
    etco2 = np.ones(30) * 5
    low = solve_ode_response_numpy(np.ones((1,)) * 0.1, np.zeros((1,)), np.ones((1,)) * 5, etco2, time)
    high = solve_ode_response_numpy(np.ones((1,)) * 0.5, np.zeros((1,)), np.ones((1,)) * 5, etco2, time)
    slow = solve_ode_response_numpy(np.ones((1,)) * 0.5, np.zeros((1,)), np.ones((1,)) * 50, etco2, time)
    assert high[-1, 0] > low[-1, 0]
    assert slow[5, 0] < high[5, 0]


def test_torch_ode_exact_update_shape_and_stability():
    torch = pytest.importorskip("torch")
    time = torch.arange(0, 12).float()
    etco2 = torch.ones(12) * 5
    cvr = torch.ones(2, 3, 4) * 0.3
    delay = torch.zeros(2, 3, 4)
    T = torch.ones(2, 3, 4) * 10
    y = simulate_ode_bold_torch(cvr, delay, T, etco2, time)
    assert y.shape == (2, 12, 3, 4)
    assert torch.isfinite(y).all()
    assert y[:, -1].mean() > y[:, 1].mean()
