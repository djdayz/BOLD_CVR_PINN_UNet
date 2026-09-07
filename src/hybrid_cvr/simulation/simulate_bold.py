from __future__ import annotations

from typing import Any

from hybrid_cvr.cvr.ode import solve_ode_response_numpy
from hybrid_cvr.simulation.co2_paradigms import block_paradigm
from hybrid_cvr.simulation.noise import add_gaussian_noise


def simulate_bold_psc(
    cvr_map: Any,
    delay_map: Any,
    T_map: Any,
    time_grid: Any,
    etco2: Any | None = None,
    noise_sigma: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    np = __import__("numpy")
    if etco2 is None:
        etco2 = block_paradigm(time_grid)
    y = solve_ode_response_numpy(cvr_map, delay_map, T_map, etco2, time_grid)
    if noise_sigma:
        y = add_gaussian_noise(y, noise_sigma, seed)
    baseline = np.ones_like(y) * 1000.0
    bold = baseline * (1.0 + y / 100.0)
    return {"bold_psc": y.astype("float32"), "bold_intensity": bold.astype("float32"), "etco2": etco2}
