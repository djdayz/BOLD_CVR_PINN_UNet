import pytest

torch = pytest.importorskip("torch")

from hybrid_cvr.pinn.delay_interp import interpolate_delayed_1d
from hybrid_cvr.pinn.torch_ode import simulate_ode_bold_torch


def test_differentiable_delay_interpolation_and_gradients():
    time = torch.arange(0, 8).float()
    etco2 = torch.linspace(0, 7, 8)
    delay = torch.ones(1, 2, 2, requires_grad=True)
    shifted = interpolate_delayed_1d(etco2, time, delay)
    shifted.sum().backward()
    assert delay.grad is not None
    assert shifted.shape == (1, 8, 2, 2)


def test_delay_interpolation_zero_pads_out_of_range_and_fractional_values():
    time = torch.arange(0, 5).float()
    etco2 = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0])
    delay = torch.full((1, 1, 1), 0.5)
    shifted = interpolate_delayed_1d(etco2, time, delay)
    assert torch.isclose(shifted[0, 0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(shifted[0, 1, 0, 0], torch.tensor(5.0))
    assert torch.isfinite(shifted).all()


def test_gradient_flow_through_cvr_delay_T():
    time = torch.arange(0, 8).float()
    etco2 = torch.linspace(0, 7, 8)
    cvr = torch.full((1, 2, 2), 0.3, requires_grad=True)
    delay = torch.ones(1, 2, 2, requires_grad=True)
    T = torch.full((1, 2, 2), 5.0, requires_grad=True)
    y = simulate_ode_bold_torch(cvr, delay, T, etco2, time)
    y.sum().backward()
    assert cvr.grad is not None
    assert delay.grad is not None
    assert T.grad is not None
