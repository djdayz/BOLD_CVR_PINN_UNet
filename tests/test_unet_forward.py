import pytest

torch = pytest.importorskip("torch")

from hybrid_cvr.models.cnn1d_unet3d_physiology import CNN1DUNet3DPhysiologyModel


def test_cnn1d_unet3d_physiology_forward_with_direct_bounded_heads():
    model = CNN1DUNet3DPhysiologyModel(
        in_channels=4,
        base_channels=4,
        depth=2,
        temporal_embedding_channels=8,
    )
    features = torch.randn(1, 4, 8, 8, 8)
    bold = torch.randn(1, 32, 8, 8, 8)
    etco2 = torch.sin(torch.arange(32).float() / 4).unsqueeze(0)
    time = torch.arange(32).float()
    mask = torch.ones(1, 8, 8, 8)
    out = model(features, etco2=etco2, time_grid=time, mask=mask, bold_psc=bold)
    assert out["cvr"].shape == (1, 8, 8, 8)
    assert out["bold_psc_hat"].shape == (1, 32, 8, 8, 8)
    assert torch.all((out["cvr"] >= 0.0) & (out["cvr"] <= 1.8))
    assert torch.all((out["delay"] >= 0.0) & (out["delay"] <= 80.0))
    assert torch.all((out["T"] >= 2.0) & (out["T"] <= 100.0))
    assert set(out["parameter_log_var"]) == {"cvr", "delay", "T"}
    assert out["cvr_profile"].shape == out["cvr"].shape
    assert out["cvr_correction_gate"].shape == out["cvr"].shape
    assert torch.all(out["cvr_correction_factor"] > 0)


def test_cvr_residual_head_starts_at_physical_profile_and_receives_gradients():
    model = CNN1DUNet3DPhysiologyModel(
        in_channels=4,
        base_channels=4,
        depth=1,
        temporal_embedding_channels=8,
        cvr_residual_channels=8,
        initial_parameter_values={"delay": 2.0, "T": 5.0},
    )
    out = model(
        torch.randn(1, 4, 8, 8, 8),
        etco2=torch.sin(torch.arange(16).float() / 3).unsqueeze(0),
        time_grid=torch.arange(16).float(),
        mask=torch.ones(1, 8, 8, 8),
        bold_psc=torch.randn(1, 16, 8, 8, 8),
    )
    assert torch.allclose(out["cvr"], out["cvr_profile"], atol=1e-6)
    out["bold_psc_hat"].square().mean().backward()
    final = model.cvr_residual_head.net[-1]
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()


def test_temporal_hybrid_joint_head_uses_no_tissue_input():
    model = CNN1DUNet3DPhysiologyModel(
        in_channels=10,
        base_channels=4,
        depth=1,
        temporal_embedding_channels=8,
        joint_parameter_head=True,
    )
    features = torch.randn(1, 10, 8, 8, 8)
    bold = torch.randn(1, 16, 8, 8, 8)
    etco2 = torch.sin(torch.arange(16).float() / 3).unsqueeze(0)
    out = model(
        features,
        etco2=etco2,
        time_grid=torch.arange(16).float(),
        mask=torch.ones(1, 8, 8, 8),
        tissue_maps=None,
        bold_psc=bold,
    )
    assert out["cvr"].shape == (1, 8, 8, 8)
    assert out["bold_psc_hat"].shape == (1, 16, 8, 8, 8)
    assert model.joint_head.net[-1].out_channels == 6


def test_temporal_hybrid_uses_physiological_initial_values():
    model = CNN1DUNet3DPhysiologyModel(
        in_channels=10,
        base_channels=4,
        depth=1,
        temporal_embedding_channels=8,
        joint_parameter_head=True,
        initial_parameter_values={"delay": 25.0, "T": 25.0},
    )
    out = model(
        torch.zeros(1, 10, 8, 8, 8),
        etco2=torch.zeros(1, 16),
        time_grid=torch.arange(16).float(),
        mask=torch.ones(1, 8, 8, 8),
        bold_psc=torch.zeros(1, 16, 8, 8, 8),
    )
    assert out["delay"].mean().item() == pytest.approx(25.0, abs=1.0)
    assert out["T"].mean().item() == pytest.approx(25.0, abs=1.0)
