import pytest

torch = pytest.importorskip("torch")

from hybrid_cvr.models.hybrid_unet_pinn import HybridUNetPINN


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
