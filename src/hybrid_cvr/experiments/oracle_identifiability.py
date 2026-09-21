from __future__ import annotations

from dataclasses import dataclass, field
import csv
import itertools
import json
from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency
from hybrid_cvr.cvr.ode import solve_ode_response_numpy
from hybrid_cvr.simulation.co2_paradigms import make_paradigm


@dataclass(frozen=True)
class OracleIdentifiabilityConfig:
    output_dir: Path = Path("data/qc/oracle_identifiability")
    paradigms: tuple[str, ...] = ("block", "multi_step", "pseudo_random_binary")
    tcnr_levels: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    noise_seeds: tuple[int, ...] = (17, 29, 41)
    cvr_values: tuple[float, ...] = (0.10, 0.20, 0.50)
    delay_values: tuple[float, ...] = (5.0, 20.0, 45.0)
    T_values: tuple[float, ...] = (15.0, 35.0, 60.0)
    n_timepoints: int = 480
    tr_seconds: float = 1.55
    fit_steps: int = 450
    fit_lr: float = 0.05
    huber_delta: float = 1.0
    cvr_bounds: tuple[float, float] = (-0.4, 1.8)
    delay_bounds: tuple[float, float] = (0.0, 80.0)
    T_bounds: tuple[float, float] = (2.0, 100.0)
    profile_limit: int = 36
    profile_T_grid: tuple[float, ...] = tuple(float(v) for v in range(5, 86, 5))
    profile_delay_grid: tuple[float, ...] = tuple(float(v) for v in range(0, 81, 5))
    plot: bool = True
    seed: int = 17
    progress_every: int = 25


def run_oracle_identifiability(config: OracleIdentifiabilityConfig) -> dict[str, Path | int]:
    np = require_dependency("numpy", "pip install numpy")
    scipy_stats = require_dependency("scipy.stats", "pip install scipy")

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    time_grid = np.arange(int(config.n_timepoints), dtype=np.float32) * float(config.tr_seconds)
    rows: list[dict[str, Any]] = []
    profile_T_rows: list[dict[str, Any]] = []
    profile_delay_rows: list[dict[str, Any]] = []
    profile_cases: list[dict[str, Any]] = []
    combos = list(
        itertools.product(
            config.paradigms,
            config.tcnr_levels,
            config.noise_seeds,
            config.cvr_values,
            config.delay_values,
            config.T_values,
        )
    )
    if config.profile_limit > 0:
        profile_indices = set(
            int(round(v))
            for v in np.linspace(0, max(len(combos) - 1, 0), min(int(config.profile_limit), len(combos)))
        )
    else:
        profile_indices = set()
    for case_index, (paradigm, tcnr, noise_seed, cvr, delay, T) in enumerate(combos):
        etco2 = make_paradigm(str(paradigm), time_grid, seed=int(config.seed) + case_index).astype(
            np.float32
        )
        clean = _simulate_voxel(float(cvr), float(delay), float(T), etco2, time_grid)
        observed, noise_sigma = _add_tcnr_noise(clean, float(tcnr), int(noise_seed))
        fit = fit_oracle_voxel(observed, etco2, time_grid, config)
        row = {
            "case_index": case_index,
            "paradigm": str(paradigm),
            "tcnr": float(tcnr),
            "noise_seed": int(noise_seed),
            "true_cvr": float(cvr),
            "true_delay": float(delay),
            "true_T": float(T),
            "pred_cvr": float(fit["cvr"]),
            "pred_delay": float(fit["delay"]),
            "pred_T": float(fit["T"]),
            "loss": float(fit["loss"]),
            "noise_sigma": float(noise_sigma),
            "cvr_error": float(fit["cvr"] - cvr),
            "delay_error": float(fit["delay"] - delay),
            "T_error": float(fit["T"] - T),
        }
        rows.append(row)
        if config.progress_every > 0 and (
            len(rows) == 1 or len(rows) % int(config.progress_every) == 0 or len(rows) == len(combos)
        ):
            print(f"[oracle] fitted {len(rows)}/{len(combos)} cases", flush=True)
        if case_index in profile_indices:
            profile_cases.append(
                {
                    **row,
                    "etco2": etco2,
                    "observed": observed,
                    "clean": clean,
                }
            )

    for case in profile_cases:
        profile_T_rows.extend(_profile_fixed_T(case, time_grid, config))
        profile_delay_rows.extend(_profile_fixed_delay(case, time_grid, config))

    predictions_path = out / "oracle_predictions.csv"
    metrics_path = out / "oracle_metrics.csv"
    profile_T_path = out / "profile_fixed_T.csv"
    profile_delay_path = out / "profile_fixed_delay.csv"
    summary_path = out / "oracle_summary.json"
    _write_csv(predictions_path, rows)
    _write_csv(profile_T_path, _strip_profile_arrays(profile_T_rows))
    _write_csv(profile_delay_path, _strip_profile_arrays(profile_delay_rows))
    metrics_rows = _metrics_by_group(rows, scipy_stats)
    _write_csv(metrics_path, metrics_rows)
    summary = {
        "n_cases": len(rows),
        "n_profile_cases": len(profile_cases),
        "outputs": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "profile_fixed_T": str(profile_T_path),
            "profile_fixed_delay": str(profile_delay_path),
        },
        "config": _config_to_jsonable(config),
        "interpretation": (
            "Wide/flat profile losses indicate practical non-identifiability; "
            "large T errors with flat T profiles should be interpreted as uncertainty, "
            "not as a neural-network-only failure."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.plot:
        _write_plots(out, rows, profile_T_rows, profile_delay_rows)
    return {
        "output_dir": out,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "profile_fixed_T": profile_T_path,
        "profile_fixed_delay": profile_delay_path,
        "summary": summary_path,
        "n_cases": len(rows),
    }


def fit_oracle_voxel(y_obs: Any, etco2: Any, time_grid: Any, config: OracleIdentifiabilityConfig) -> dict[str, float]:
    np = require_dependency("numpy", "pip install numpy")
    y = np.asarray(y_obs, dtype=np.float64)
    starts = (
        (0.20, 15.0, 25.0),
        (0.35, 35.0, 45.0),
        (0.60, 55.0, 70.0),
    )
    bounds = (config.cvr_bounds, config.delay_bounds, config.T_bounds)

    def objective(x: Any) -> float:
        cvr, delay, T = [float(v) for v in x]
        y_hat = _simulate_voxel(cvr, delay, T, etco2, time_grid).astype(np.float64)
        return _huber_mean(y_hat - y, float(config.huber_delta))

    fit = _minimize_with_starts(objective, starts, bounds)
    return {
        "loss": float(fit["loss"]),
        "cvr": float(fit["x"][0]),
        "delay": float(fit["x"][1]),
        "T": float(fit["x"][2]),
    }


def _simulate_voxel(cvr: float, delay: float, T: float, etco2: Any, time_grid: Any):
    np = require_dependency("numpy", "pip install numpy")
    y = solve_ode_response_numpy(
        np.asarray([cvr], dtype=np.float32),
        np.asarray([delay], dtype=np.float32),
        np.asarray([T], dtype=np.float32),
        etco2,
        time_grid,
    )
    return np.asarray(y[:, 0], dtype=np.float32)


def _add_tcnr_noise(clean: Any, tcnr: float, seed: int):
    np = require_dependency("numpy", "pip install numpy")
    rng = np.random.default_rng(seed)
    clean = np.asarray(clean, dtype=np.float32)
    signal_scale = float(np.std(clean))
    if signal_scale <= 0:
        signal_scale = max(float(np.max(np.abs(clean))), 1.0)
    sigma = signal_scale / max(float(tcnr), 1e-6)
    noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    return (clean + noise).astype(np.float32), sigma


def _profile_fixed_T(case: dict[str, Any], time_grid: Any, config: OracleIdentifiabilityConfig) -> list[dict[str, Any]]:
    rows = []
    for T_fixed in config.profile_T_grid:
        fit = _fit_two_params(
            case["observed"],
            case["etco2"],
            time_grid,
            config,
            fixed_name="T",
            fixed_value=float(T_fixed),
        )
        rows.append(
            {
                "case_index": case["case_index"],
                "paradigm": case["paradigm"],
                "tcnr": case["tcnr"],
                "true_T": case["true_T"],
                "fixed_T": float(T_fixed),
                **fit,
            }
        )
    return _add_profile_delta_and_width(rows, "fixed_T")


def _profile_fixed_delay(case: dict[str, Any], time_grid: Any, config: OracleIdentifiabilityConfig) -> list[dict[str, Any]]:
    rows = []
    for delay_fixed in config.profile_delay_grid:
        fit = _fit_two_params(
            case["observed"],
            case["etco2"],
            time_grid,
            config,
            fixed_name="delay",
            fixed_value=float(delay_fixed),
        )
        rows.append(
            {
                "case_index": case["case_index"],
                "paradigm": case["paradigm"],
                "tcnr": case["tcnr"],
                "true_delay": case["true_delay"],
                "fixed_delay": float(delay_fixed),
                **fit,
            }
        )
    return _add_profile_delta_and_width(rows, "fixed_delay")


def _fit_two_params(
    y_obs: Any,
    etco2: Any,
    time_grid: Any,
    config: OracleIdentifiabilityConfig,
    *,
    fixed_name: str,
    fixed_value: float,
) -> dict[str, float]:
    np = require_dependency("numpy", "pip install numpy")
    y = np.asarray(y_obs, dtype=np.float64)
    if fixed_name == "T":
        starts = ((0.20, 15.0), (0.35, 35.0), (0.60, 55.0))
        bounds = (config.cvr_bounds, config.delay_bounds)

        def unpack(x: Any) -> tuple[float, float, float]:
            return float(x[0]), float(x[1]), float(fixed_value)

    elif fixed_name == "delay":
        starts = ((0.20, 25.0), (0.35, 45.0), (0.60, 70.0))
        bounds = (config.cvr_bounds, config.T_bounds)

        def unpack(x: Any) -> tuple[float, float, float]:
            return float(x[0]), float(fixed_value), float(x[1])

    else:
        raise ValueError("fixed_name must be T or delay")

    def objective(x: Any) -> float:
        cvr, delay, T = unpack(x)
        y_hat = _simulate_voxel(cvr, delay, T, etco2, time_grid).astype(np.float64)
        return _huber_mean(y_hat - y, float(config.huber_delta))

    fit = _minimize_with_starts(objective, starts, bounds)
    cvr, delay, T = unpack(fit["x"])
    return {"loss": float(fit["loss"]), "cvr": cvr, "delay": delay, "T": T}


def _huber_mean(residual: Any, delta: float) -> float:
    np = require_dependency("numpy", "pip install numpy")
    r = np.asarray(residual, dtype=np.float64)
    abs_r = np.abs(r)
    d = max(float(delta), 1e-12)
    loss = np.where(abs_r <= d, 0.5 * r * r, d * (abs_r - 0.5 * d))
    return float(np.mean(loss))


def _minimize_with_starts(objective: Any, starts: tuple[tuple[float, ...], ...], bounds: Any) -> dict[str, Any]:
    scipy_optimize = require_dependency("scipy.optimize", "pip install scipy")
    best_x = None
    best_loss = float("inf")
    for start in starts:
        result = scipy_optimize.minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 120, "ftol": 1e-9},
        )
        loss = float(result.fun)
        if loss < best_loss:
            best_loss = loss
            best_x = result.x
    return {"loss": best_loss, "x": best_x}


def _add_profile_delta_and_width(rows: list[dict[str, Any]], fixed_key: str) -> list[dict[str, Any]]:
    if not rows:
        return rows
    min_loss = min(float(r["loss"]) for r in rows)
    threshold = min_loss + max(0.01 * abs(min_loss), 1e-4)
    near = [float(r[fixed_key]) for r in rows if float(r["loss"]) <= threshold]
    width = max(near) - min(near) if near else 0.0
    for row in rows:
        row["profile_delta_loss"] = float(row["loss"]) - min_loss
        row["near_min_width"] = width
    return rows


def _metrics_by_group(rows: list[dict[str, Any]], scipy_stats: Any) -> list[dict[str, Any]]:
    out = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {("all", "all", "all"): rows}
    for key in ("paradigm", "tcnr"):
        values = sorted({str(r[key]) for r in rows})
        for value in values:
            groups[(key, value, "all")] = [r for r in rows if str(r[key]) == value]
    for (group_key, group_value, _), group_rows in groups.items():
        for param in ("cvr", "delay", "T"):
            true = [float(r[f"true_{param}"]) for r in group_rows]
            pred = [float(r[f"pred_{param}"]) for r in group_rows]
            err = [p - t for p, t in zip(pred, true, strict=True)]
            out.append(
                {
                    "group_key": group_key,
                    "group_value": group_value,
                    "parameter": param,
                    "n": len(group_rows),
                    "rmse": _rmse(err),
                    "mae": _mae(err),
                    "bias": sum(err) / max(len(err), 1),
                    "pearson": _corr(scipy_stats.pearsonr, true, pred),
                    "spearman": _corr(scipy_stats.spearmanr, true, pred),
                }
            )
    for a, b in (("cvr", "delay"), ("cvr", "T"), ("delay", "T")):
        ea = [float(r[f"{a}_error"]) for r in rows]
        eb = [float(r[f"{b}_error"]) for r in rows]
        out.append(
            {
                "group_key": "error_correlation",
                "group_value": f"{a}_vs_{b}",
                "parameter": "error_pair",
                "n": len(rows),
                "rmse": "",
                "mae": "",
                "bias": "",
                "pearson": _corr(scipy_stats.pearsonr, ea, eb),
                "spearman": _corr(scipy_stats.spearmanr, ea, eb),
            }
        )
    return out


def _rmse(err: list[float]) -> float:
    np = require_dependency("numpy", "pip install numpy")
    return float(np.sqrt(np.mean(np.asarray(err, dtype=float) ** 2))) if err else float("nan")


def _mae(err: list[float]) -> float:
    np = require_dependency("numpy", "pip install numpy")
    return float(np.mean(np.abs(np.asarray(err, dtype=float)))) if err else float("nan")


def _corr(fn, a: list[float], b: list[float]) -> float:
    np = require_dependency("numpy", "pip install numpy")
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size < 2 or float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return float("nan")
    result = fn(aa, bb)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _strip_profile_arrays(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k not in {"etco2", "observed", "clean"}} for row in rows]


def _config_to_jsonable(config: OracleIdentifiabilityConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.__dict__.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _write_plots(out: Path, rows: list[dict[str, Any]], profile_T_rows: list[dict[str, Any]], profile_delay_rows: list[dict[str, Any]]) -> None:
    np = require_dependency("numpy", "pip install numpy")
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    for param in ("cvr", "delay", "T"):
        fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=150)
        true = np.asarray([r[f"true_{param}"] for r in rows], dtype=float)
        pred = np.asarray([r[f"pred_{param}"] for r in rows], dtype=float)
        ax.scatter(true, pred, s=10, alpha=0.55)
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel(f"True {param}")
        ax.set_ylabel(f"Recovered {param}")
        fig.tight_layout()
        fig.savefig(out / f"oracle_true_vs_pred_{param}.png")
        plt.close(fig)
    _plot_profile_examples(out, profile_T_rows, "fixed_T", "T")
    _plot_profile_examples(out, profile_delay_rows, "fixed_delay", "delay")


def _plot_profile_examples(out: Path, rows: list[dict[str, Any]], fixed_key: str, label: str) -> None:
    if not rows:
        return
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["case_index"]), []).append(row)
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=150)
    for _, group_rows in list(grouped.items())[:6]:
        x = [float(r[fixed_key]) for r in group_rows]
        y = [float(r["profile_delta_loss"]) for r in group_rows]
        ax.plot(x, y, linewidth=1)
    ax.set_xlabel(f"Fixed {label}")
    ax.set_ylabel("Loss - min loss")
    fig.tight_layout()
    fig.savefig(out / f"profile_fixed_{label}.png")
    plt.close(fig)
