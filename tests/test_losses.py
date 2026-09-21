import pytest

torch = pytest.importorskip("torch")

from hybrid_cvr.physiology.losses import (
    LossWeights,
    assert_no_supervised_parameter_loss,
    self_supervised_loss,
)


def test_no_supervised_parameter_loss_config_is_rejected():
    with pytest.raises(ValueError):
        assert_no_supervised_parameter_loss({}, {"supervised_cvr": 1.0})


def test_losses_reject_gt_maps_in_batch():
    pred = {
        "bold_psc_hat": torch.zeros(1, 5, 4, 4),
        "cvr": torch.ones(1, 4, 4) * 0.2,
        "delay": torch.ones(1, 4, 4),
        "T": torch.ones(1, 4, 4) * 10,
        "sigma_y": torch.ones(1, 4, 4),
        "raw": {name: torch.zeros(1, 4, 4) for name in ("cvr", "delay", "T")},
        "nuisance_coefficients": torch.zeros(1, 2, 4, 4),
    }
    batch = {
        "bold_psc": torch.zeros(1, 5, 4, 4),
        "etco2_for_model": torch.ones(1, 5),
        "time_grid": torch.arange(5).float(),
        "mask": torch.ones(1, 4, 4),
        "GT_CVR": torch.ones(1, 4, 4) * 99,
        "GT_delay": torch.ones(1, 4, 4) * 99,
        "GT_T": torch.ones(1, 4, 4) * 99,
    }
    with pytest.raises(ValueError):
        self_supervised_loss(pred, batch)


def test_stage_two_losses_are_finite():
    pred = {
        "bold_psc_hat": torch.zeros(1, 32, 2, 2),
        "cvr": torch.ones(1, 2, 2) * 0.2,
        "delay": torch.ones(1, 2, 2),
        "T": torch.ones(1, 2, 2) * 10,
        "sigma_y": torch.ones(1, 2, 2),
        "raw": {name: torch.zeros(1, 2, 2) for name in ("cvr", "delay", "T")},
        "nuisance_coefficients": torch.zeros(1, 2, 2, 2),
    }
    batch = {
        "bold_psc": torch.randn(1, 32, 2, 2) * 0.1,
        "etco2_for_model": torch.sin(2 * torch.pi * torch.arange(32).float() / 8).unsqueeze(0),
        "time_grid": torch.arange(32).float(),
        "mask": torch.ones(1, 2, 2),
    }
    losses = self_supervised_loss(
        pred,
        batch,
        LossWeights(lambda_data=1.0, lambda_weak=0.1),
    )
    assert torch.isfinite(losses["data"])
    assert torch.isfinite(losses["weak"])
    assert torch.isfinite(losses["total"])
