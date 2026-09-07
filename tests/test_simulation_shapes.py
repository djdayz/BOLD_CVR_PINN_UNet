import pytest

np = pytest.importorskip("numpy")

from hybrid_cvr.simulation.phantom import default_parameter_maps, make_phantom_labels
from hybrid_cvr.simulation.simulate_bold import simulate_bold_psc


def test_simulation_output_shapes():
    labels = make_phantom_labels((12, 12, 3))
    maps = default_parameter_maps(labels)
    time = np.arange(10, dtype=float)
    z = 1
    sim = simulate_bold_psc(maps["GT_CVR"][..., z], maps["GT_delay"][..., z], maps["GT_T"][..., z], time)
    assert sim["bold_psc"].shape == (10, 12, 12)
    assert sim["bold_intensity"].shape == (10, 12, 12)
