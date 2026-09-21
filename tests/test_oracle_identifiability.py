from __future__ import annotations

import csv

import pytest


def test_oracle_identifiability_smoke(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from hybrid_cvr.experiments.oracle_identifiability import (
        OracleIdentifiabilityConfig,
        run_oracle_identifiability,
    )

    cfg = OracleIdentifiabilityConfig(
        output_dir=tmp_path / "oracle",
        paradigms=("block",),
        tcnr_levels=(10.0,),
        noise_seeds=(17,),
        cvr_values=(0.2,),
        delay_values=(20.0,),
        T_values=(35.0,),
        n_timepoints=180,
        fit_steps=5,
        profile_limit=0,
        plot=False,
    )
    result = run_oracle_identifiability(cfg)

    assert result["n_cases"] == 1
    assert result["predictions"].exists()
    assert result["metrics"].exists()
    assert result["summary"].exists()

    with result["predictions"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["paradigm"] == "block"
    assert float(rows[0]["loss"]) >= 0.0
