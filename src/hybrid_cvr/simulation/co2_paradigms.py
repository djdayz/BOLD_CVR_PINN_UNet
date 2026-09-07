from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def smooth_step(time: Any, onset: float, amplitude: float, ramp_seconds: float = 8.0) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    return 0.5 * amplitude * (1.0 + np.tanh((np.asarray(time) - onset) / max(ramp_seconds, 1e-6)))


def block_paradigm(
    time: Any,
    amplitude: float = 8.0,
    onsets: tuple[float, ...] = (60.0, 180.0, 300.0, 420.0),
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    u = np.zeros_like(np.asarray(time, dtype=float))
    sign = 1.0
    for onset in onsets:
        u += sign * smooth_step(time, onset, amplitude, ramp_seconds=8.0)
        sign *= -1.0
    return u


def ramp_paradigm(time: Any, amplitude: float = 10.0) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    t = np.asarray(time, dtype=float)
    return amplitude * (t - t.min()) / max(t.max() - t.min(), 1e-6)


def sinusoidal_paradigm(time: Any, amplitude: float = 4.0, period_seconds: float = 120.0) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    return amplitude * np.sin(2.0 * np.pi * np.asarray(time) / period_seconds)


def multi_step_paradigm(time: Any, levels: tuple[float, ...] = (0.0, 4.0, 8.0, 2.0, 10.0, 0.0)) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return t.copy()
    edges = np.linspace(float(t.min()), float(t.max()), len(levels) + 1)
    u = np.zeros_like(t)
    previous = float(levels[0])
    for idx, level in enumerate(levels[1:], start=1):
        onset = edges[idx]
        u += smooth_step(t, onset, float(level) - previous, ramp_seconds=10.0)
        previous = float(level)
    return u


def pseudo_random_binary_paradigm(
    time: Any,
    amplitude: float = 7.0,
    block_seconds: float = 45.0,
    seed: int | None = None,
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return t.copy()
    rng = np.random.default_rng(seed)
    n_blocks = max(2, int(np.ceil((t.max() - t.min()) / max(block_seconds, 1e-6))))
    states = rng.integers(0, 2, size=n_blocks + 1).astype(float) * amplitude
    states[0] = 0.0
    edges = t.min() + np.arange(n_blocks + 1) * block_seconds
    u = np.zeros_like(t)
    previous = states[0]
    for onset, state in zip(edges[1:], states[1:], strict=False):
        u += smooth_step(t, float(onset), float(state - previous), ramp_seconds=8.0)
        previous = state
    return u


def breath_hold_like_paradigm(
    time: Any,
    amplitude: float = 7.0,
    hold_seconds: float = 25.0,
    period_seconds: float = 120.0,
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return t.copy()
    u = np.zeros_like(t)
    first_onset = float(t.min()) + 70.0
    onsets = np.arange(first_onset, float(t.max()), period_seconds)
    for onset in onsets:
        u += smooth_step(t, float(onset), amplitude, ramp_seconds=6.0)
        u -= smooth_step(t, float(onset + hold_seconds), amplitude, ramp_seconds=12.0)
    return u


def resting_state_like_paradigm(
    time: Any,
    amplitude: float = 2.0,
    seed: int | None = None,
    smoothing_seconds: float = 35.0,
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return t.copy()
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=t.size)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    sigma = max(smoothing_seconds / max(dt, 1e-6), 1.0)
    smooth = ndi.gaussian_filter1d(x, sigma=sigma, mode="nearest")
    smooth -= float(np.mean(smooth))
    scale = float(np.std(smooth))
    if scale > 0:
        smooth = smooth / scale
    return amplitude * smooth


def make_paradigm(name: str, time: Any, seed: int | None = None) -> Any:
    key = name.lower().replace("-", "_")
    if key == "block":
        return block_paradigm(time)
    if key == "ramp":
        return ramp_paradigm(time)
    if key in {"multi_step", "multistep"}:
        return multi_step_paradigm(time)
    if key in {"pseudo_random_binary", "prbs"}:
        return pseudo_random_binary_paradigm(time, seed=seed)
    if key in {"sinusoidal", "sine"}:
        return sinusoidal_paradigm(time)
    if key in {"breath_hold", "breath_hold_like"}:
        return breath_hold_like_paradigm(time)
    if key in {"resting_state", "resting_state_like", "spontaneous"}:
        return resting_state_like_paradigm(time, seed=seed)
    raise ValueError(f"Unknown CO2 paradigm: {name}")
