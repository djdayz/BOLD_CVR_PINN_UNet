import pytest

torch = pytest.importorskip("torch")

from hybrid_cvr.pinn.losses import assert_no_supervised_parameter_loss, self_supervised_loss


def test_no_supervised_parameter_loss_config_is_rejected():
    with pytest.raises(ValueError):
        assert_no_supervised_parameter_loss({}, {"supervised_cvr": 1.0})


def test_losses_return_finite_values_and_ignore_gt_maps():
    pred = {
        "bold_psc_hat": torch.zeros(1, 5, 4, 4),
        "cvr": torch.ones(1, 4, 4) * 0.2,
        "delay": torch.ones(1, 4, 4),
        "T": torch.ones(1, 4, 4) * 10,
        "sigma": torch.ones(1, 4, 4),
    }
    batch = {
        "bold_psc": torch.zeros(1, 5, 4, 4),
        "etco2": torch.ones(5),
        "time_grid": torch.arange(5).float(),
        "mask": torch.ones(1, 4, 4),
        "GT_CVR": torch.ones(1, 4, 4) * 99,
        "GT_delay": torch.ones(1, 4, 4) * 99,
        "GT_T": torch.ones(1, 4, 4) * 99,
    }
    losses = self_supervised_loss(pred, batch)
    assert torch.isfinite(losses["total"])
