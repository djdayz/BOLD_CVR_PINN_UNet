import pytest

torch = pytest.importorskip("torch")

from hybrid_cvr.models.hybrid_unet_pinn import HybridUNetPINN
from hybrid_cvr.models.temporal_hybrid_3d import TemporalHybridUNetPINN


def test_unet_output_shape_and_constraints():
    model = HybridUNetPINN(in_channels=3, base_channels=4, depth=2, T_mode="voxelwise")
    x = torch.randn(2, 3, 16, 16)
    maps = model(x)
    assert maps["cvr"].min() >= -0.4
    assert maps["cvr"].max() <= 1.8
    assert maps["delay"].min() >= 0
    assert maps["delay"].max() <= 80
    assert maps["T"].min() >= 2
    assert maps["T"].max() <= 100


def test_unet_T_modes_forward_with_ode():
    time = torch.arange(0, 8).float()
    etco2 = torch.ones(2, 8)
    x = torch.randn(2, 3, 16, 16)
    mask = torch.ones(2, 16, 16)
    tissue = torch.zeros(2, 5, 16, 16)
    tissue[:, 0] = 1.0
    for mode in ("global", "tissuewise", "voxelwise"):
        model = HybridUNetPINN(in_channels=3, base_channels=4, depth=2, T_mode=mode)
        out = model(x, etco2=etco2, time_grid=time, mask=mask, tissue_maps=tissue)
        assert out["T"].shape == (2, 16, 16)
        assert out["bold_psc_hat"].shape == (2, 8, 16, 16)


def test_temporal_hybrid_3d_forward_with_direct_bounded_heads():
    model = TemporalHybridUNetPINN(
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


def test_temporal_hybrid_joint_head_uses_no_tissue_input():
    model = TemporalHybridUNetPINN(
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
    assert model.joint_head.net[-1].out_channels == 7
