from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.sloppiness import (
    analyse,
    bin_factor,
    joint_effects,
    load_factors,
    spearman_with_ci,
)

N = 600


def _manifest(rng, **overrides):
    frame = pd.DataFrame(
        {
            "query_id": [f"q{i:04d}" for i in range(N)],
            "pattern_id": [f"p{i // 2}" for i in range(N)],
            "mode": np.where(np.arange(N) % 2 == 0, "Personal", "Fixed"),
            "offset_ms": rng.uniform(-40, 40, N),
            "jitter_ms": rng.uniform(0, 20, N),
            "drift_pct": np.zeros(N),
            "drop_prob": rng.uniform(0, 0.3, N),
            "dropped_hits": rng.integers(0, 6, N),
            "snr_db": rng.uniform(10, 30, N),
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


def _system(manifest, rr):
    table = pd.DataFrame({"query_id": manifest["query_id"], "rr": rr, "ap": rr})
    return table.set_index("query_id").sort_index()


def test_binning_orders_cleanest_to_messiest_and_labels_the_real_values():
    badness = pd.Series(-np.arange(100, dtype=float), index=[f"q{i}" for i in range(100)])
    shown = -badness  # signal-to-noise: the displayed value is the raw dB
    codes, labels = bin_factor(badness, shown, 4)
    assert codes.nunique() == 4
    ## highest dB (cleanest) is bin 0, and its label shows dB values
    assert codes.iloc[-1] == 0
    assert labels[0].startswith("75") or labels[0].startswith("74")


def test_a_constant_factor_gives_no_bins_and_few_values_become_categories():
    index = [f"q{i}" for i in range(50)]
    assert bin_factor(pd.Series(0.0, index=index), pd.Series(0.0, index=index), 4) is None
    few = pd.Series([0.0, 1.0, 2.0] * 16 + [0.0, 0.0], index=index)
    codes, labels = bin_factor(few, few, 4)
    assert sorted(codes.unique()) == [0, 1, 2]
    assert labels == ["0", "1", "2"]


def test_spearman_is_strongly_negative_when_messier_means_worse():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(100), 3).astype(str)
    badness = rng.uniform(0, 1, 300)
    rr = 1 - badness + rng.normal(0, 0.05, 300)
    rho, low, high = spearman_with_ci(badness, rr, groups, n_boot=300, seed=0)
    assert rho < -0.9 and high < -0.8 and low <= rho <= high
    flat, low, high = spearman_with_ci(badness, rng.uniform(0, 1, 300), groups, 300, 0)
    assert abs(flat) < 0.2 and low < 0 < high


def test_joint_model_finds_the_one_factor_that_matters():
    rng = np.random.default_rng(1)
    manifest = _manifest(rng)
    badness = pd.DataFrame(
        {
            "offset": manifest["offset_ms"].abs().to_numpy(),
            "jitter": manifest["jitter_ms"].to_numpy(),
        },
        index=manifest["query_id"],
    )
    rr = 0.6 - 0.01 * manifest["offset_ms"].abs().to_numpy() + rng.normal(0, 0.02, N)
    groups = manifest["pattern_id"].to_numpy()
    result = joint_effects(badness, rr, groups, n_boot=200, seed=0).set_index("factor")
    assert result.loc["offset", "effect"] < -0.08
    assert result.loc["offset", "ci_high"] < 0
    assert abs(result.loc["jitter", "effect"]) < 0.02


def test_analyse_shows_the_worse_system_falling_and_the_flat_one_staying_flat(tmp_path):
    rng = np.random.default_rng(2)
    manifest = _manifest(rng)
    path = tmp_path / "q.csv"
    manifest.to_csv(path, index=False)
    falling = _system(manifest, np.clip(0.8 - 0.015 * manifest["offset_ms"].abs(), 0, 1))
    flat = _system(manifest, rng.uniform(0.2, 0.4, N))
    index = falling.index
    factors, mode = load_factors(path, index)
    groups = manifest.set_index("query_id").loc[index, "pattern_id"].to_numpy()
    bins, trends, joint = analyse(
        {"falling": falling, "flat": flat}, groups, factors, mode, 4, 200, 0
    )
    offset = bins[(bins.factor == "offset") & (bins.system == "falling")].sort_values("bin")
    assert offset["mrr"].is_monotonic_decreasing
    assert offset["mrr"].iloc[0] - offset["mrr"].iloc[-1] > 0.3
    spearman = trends.set_index(["factor", "system"])["spearman"]
    assert spearman[("offset", "falling")] < -0.9
    assert abs(spearman[("offset", "flat")]) < 0.15
    assert {"mode"} <= set(bins["factor"])
    assert "drift" not in set(bins["factor"])  # constant column is skipped
    assert set(joint["system"]) == {"falling", "flat"}


def test_missing_columns_are_skipped_not_fatal(tmp_path):
    rng = np.random.default_rng(3)
    manifest = _manifest(rng).drop(columns=["drop_prob", "mode", "dropped_hits"])
    path = tmp_path / "q.csv"
    manifest.to_csv(path, index=False)
    factors, mode = load_factors(path, pd.Index(manifest["query_id"]))
    assert mode is None
    assert "drop_prob" not in factors.columns and "offset" in factors.columns


def test_cli_end_to_end_and_error_messages(tmp_path):
    rng = np.random.default_rng(4)
    manifest = _manifest(rng)
    manifest_path = tmp_path / "q.csv"
    manifest.to_csv(manifest_path, index=False)
    for name, rr in {
        "a": np.clip(0.8 - 0.015 * manifest["offset_ms"].abs(), 0, 1),
        "b": rng.uniform(0, 0.5, N),
    }.items():
        folder = tmp_path / name
        folder.mkdir()
        full = _system(manifest, rr).assign(
            **{"hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0, "n_relevant": 2, "chance_rr": 0.01}
        )
        full.reset_index().to_csv(folder / "per_query.csv", index=False)
        (folder / "summary.json").write_text(json.dumps({"n_library": 100}))
    out = tmp_path / "out"
    repo = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.sloppiness",
            "--system",
            f"a={tmp_path / 'a'}",
            "--system",
            f"b={tmp_path / 'b'}",
            "--query-manifest",
            str(manifest_path),
            "--out",
            str(out),
            "--n-boot",
            "100",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    for name in ("bins.csv", "trends.csv", "joint.csv"):
        assert (out / name).exists()
    assert "Timing offset" in run.stdout
    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.sloppiness",
            "--system",
            f"a={tmp_path / 'nope'}",
            "--query-manifest",
            str(manifest_path),
            "--out",
            str(out),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1 and "Sloppiness analysis failed" in bad.stderr


def test_adding_the_mode_removes_a_fake_jitter_effect():
    rng = np.random.default_rng(5)
    mode = np.where(np.arange(N) % 2 == 0, "Personal", "Fixed")
    ## the Fixed queries happen to have more jitter, and Fixed is what scores higher
    jitter = np.where(mode == "Fixed", rng.uniform(15, 40, N), rng.uniform(0, 25, N))
    rr = np.where(mode == "Fixed", 0.6, 0.3) + rng.normal(0, 0.02, N)
    ids = [f"q{i:04d}" for i in range(N)]
    badness = pd.DataFrame({"jitter": jitter}, index=ids)
    groups = np.array([f"p{i // 2}" for i in range(N)])
    without = joint_effects(badness, rr, groups, n_boot=200, seed=0).set_index("factor")
    assert without.loc["jitter", "effect"] > 0.05  # looks like jitter helps
    binary = pd.DataFrame({"mode_Personal_vs_Fixed": (mode == "Personal").astype(float)}, index=ids)
    with_mode = joint_effects(badness, rr, groups, 200, 0, binary).set_index("factor")
    assert abs(with_mode.loc["jitter", "effect"]) < 0.02
    assert with_mode.loc["mode_Personal_vs_Fixed", "effect"] == pytest.approx(-0.3, abs=0.03)
