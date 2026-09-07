import pytest

np = pytest.importorskip("numpy")

from hybrid_cvr.preprocessing.gas import (
    GasConfig,
    _baseline_value,
    detect_etco2_peaks,
    interpolate_etco2_to_grid,
    percent_to_mmhg,
    process_gas_trace,
)
from hybrid_cvr.preprocessing.qc import save_etco2_qc_plot


def test_percent_to_mmhg_conversion():
    assert np.isclose(percent_to_mmhg([5.0])[0], 38.0)


def test_etco2_peak_detection_and_interpolation():
    time = np.linspace(0, 10, 201)
    co2 = 30 + 10 * np.sin(2 * np.pi * time)
    peak_time, peak_co2 = detect_etco2_peaks(time, co2, GasConfig(co2_units="mmHg"))
    target = np.arange(0, 10, 1.0)
    interp = interpolate_etco2_to_grid(peak_time, peak_co2, target)
    assert interp.shape == target.shape
    assert np.all(np.isfinite(interp))


def test_save_etco2_qc_plot(tmp_path):
    pytest.importorskip("matplotlib")
    raw = tmp_path / "raw_gas.tsv"
    peaks = tmp_path / "etco2_peaks.tsv"
    resampled = tmp_path / "etco2_resampled.tsv"
    raw.write_text("time\tco2_mmhg\n0\t1\n1\t38\n2\t2\n3\t40\n", encoding="utf-8")
    peaks.write_text("time\tetco2_peak_mmhg\n1\t38\n3\t40\n", encoding="utf-8")
    resampled.write_text(
        "time\tetco2_mmhg\tdelta_etco2_mmhg\tbaseline_mmhg\n0\t38\t0\t38\n1\t39\t1\t38\n",
        encoding="utf-8",
    )

    out = save_etco2_qc_plot(raw, peaks, resampled, tmp_path / "etco2_qc.png")

    assert out.exists()
    assert out.stat().st_size > 0


def test_baseline_uses_first_peak_window_by_default():
    time = np.arange(0, 8, dtype=float)
    etco2 = np.asarray([99, 99, 30, 34, 36, 50, 55, 60], dtype=float)
    baseline = _baseline_value(
        time,
        etco2,
        peak_time=np.asarray([2.0, 3.0, 4.0], dtype=float),
        config=GasConfig(baseline_first_seconds=2.0),
    )

    assert baseline == pytest.approx(np.mean([30, 34, 36]))


def test_process_gas_trace_uses_source_time_offset(tmp_path):
    gas = tmp_path / "gas.tsv"
    gas.write_text(
        "sec\tPctCO2\tTime\n"
        "0\t30\t12:00:00\n"
        "10\t50\t12:00:10\n"
        "20\t30\t12:00:20\n"
        "30\t60\t12:00:30\n"
        "40\t30\t12:00:40\n",
        encoding="utf-8",
    )

    result = process_gas_trace(
        gas,
        np.asarray([0.0, 10.0]),
        GasConfig(
            time_column="sec",
            co2_column="PctCO2",
            co2_units="mmHg",
            smoothing_seconds=0.0,
            baseline_reference="target_start",
            baseline_first_seconds=1.0,
        ),
        source_time_offset_seconds=20.0,
    )

    assert np.allclose(result["source_time"], [20.0, 30.0])
    assert np.allclose(result["etco2_mmhg"], [55.0, 60.0])
