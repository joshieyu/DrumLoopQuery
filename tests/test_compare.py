from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.compare import (
    align,
    chance_hit,
    compare,
    groups_from_manifest,
    holm,
    latex_table,
    load_system,
    paired_comparison,
    parse_systems,
    participant_wins,
)
from src.eval.evaluate import EvalError


def _table(rr, n_relevant=2):
    rr = np.asarray(rr, dtype=float)
    return pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(len(rr))],
            "rr": rr,
            "ap": rr,
            "hit@1": (rr == 1.0).astype(float),
            "hit@5": (rr >= 0.2).astype(float),
            "hit@10": (rr >= 0.1).astype(float),
            "n_relevant": n_relevant,
            "chance_rr": 0.01,
        }
    ).set_index("query_id")


def test_chance_hit_matches_a_hand_calculation():
    ## 2 relevant of 10, top 1: 2/10. Top 2: 1 - C(8,2)/C(10,2) = 1 - 28/45
    assert chance_hit(10, 2, 1) == pytest.approx(0.2)
    assert chance_hit(10, 2, 2) == pytest.approx(1 - 28 / 45)
    assert chance_hit(10, 2, 10) == 1.0
    assert chance_hit(1786, 2, 10) == pytest.approx(0.0112, abs=0.0001)


def test_holm_matches_a_hand_example_and_stays_ordered():
    adjusted = holm({("a", "b"): 0.01, ("a", "c"): 0.04, ("b", "c"): 0.03})
    ## sorted p: 0.01 x3 = 0.03, 0.03 x2 = 0.06, 0.04 x1 = 0.04 -> raised to 0.06
    assert adjusted[("a", "b")] == pytest.approx(0.03)
    assert adjusted[("b", "c")] == pytest.approx(0.06)
    assert adjusted[("a", "c")] == pytest.approx(0.06)
    assert holm({("a", "b"): 0.9})[("a", "b")] == 0.9
    assert holm({("a", "b"): 0.6, ("a", "c"): 0.7})[("a", "b")] == 1.0


def test_a_clearly_better_system_is_significant_and_a_copy_is_not():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 3).astype(str)
    base = rng.uniform(0.0, 0.5, 180)
    better = base + 0.2
    result = paired_comparison(better, base, groups, n_boot=2000, n_perm=4000)
    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["ci_low"] > 0.15
    assert result["p_value"] < 0.01
    assert result["n_groups"] == 60
    same = paired_comparison(base, base, groups, n_boot=500, n_perm=2000)
    assert same["mean_difference"] == 0.0
    assert same["p_value"] == 1.0


def test_noise_around_zero_is_not_significant():
    rng = np.random.default_rng(1)
    groups = np.arange(200).astype(str)
    a = rng.uniform(0, 1, 200)
    b = a + rng.normal(0, 0.1, 200)
    assert paired_comparison(a, b, groups, n_boot=1000, n_perm=4000)["p_value"] > 0.05


def test_five_groups_can_never_reach_significance():
    ## with 5 groups there are only 32 sign patterns, so the smallest p-value is about 1/16
    groups = np.repeat(np.arange(5), 40).astype(str)
    a = np.ones(200)
    result = paired_comparison(a, np.zeros(200), groups, n_boot=500, n_perm=20000)
    assert result["p_value"] > 0.05


def test_participant_wins_counts_each_person_once():
    a = np.array([0.9, 0.9, 0.1, 0.1, 0.5])
    b = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    who = np.array(["P1", "P1", "P2", "P2", "P3"])
    assert participant_wins(a, b, who) == (1, 1, 1)


def test_systems_scored_on_different_queries_are_rejected():
    tables = {"a": _table([0.5, 0.5]), "b": _table([0.5, 0.5, 0.5])}
    with pytest.raises(EvalError, match="different queries"):
        align(tables)


def test_group_lookup_uses_the_requested_column(tmp_path):
    manifest = tmp_path / "q.csv"
    manifest.write_text("query_id,pattern_id,participant_id\nq0,A,P1\nq1,A,P2\nq2,,P2\n")
    index = pd.Index(["q0", "q1", "q2", "q3"])
    assert groups_from_manifest(index, manifest, "pattern_id").tolist() == ["A", "A", "q2", "q3"]
    assert groups_from_manifest(index, manifest, "participant_id").tolist() == [
        "P1",
        "P2",
        "P2",
        "q3",
    ]
    with pytest.raises(EvalError, match="needs query_id and nope"):
        groups_from_manifest(index, manifest, "nope")


def test_system_argument_parsing():
    assert parse_systems(["a=x/y", "b=z"]) == {"a": Path("x/y"), "b": Path("z")}
    with pytest.raises(EvalError, match="NAME=path"):
        parse_systems(["nopath", "b=z"])
    with pytest.raises(EvalError, match="at least 2 systems"):
        parse_systems(["a=x"])


def test_loading_a_results_folder_reads_the_library_size(tmp_path):
    folder = tmp_path / "sys"
    folder.mkdir()
    _table([0.5, 1.0]).reset_index().to_csv(folder / "per_query.csv", index=False)
    (folder / "summary.json").write_text('{"n_library": 1786}')
    table, n_library = load_system(folder)
    assert n_library == 1786
    assert list(table.index) == ["q0", "q1"]
    bare = tmp_path / "bare.csv"
    _table([0.5]).reset_index().to_csv(bare, index=False)
    assert load_system(bare)[1] is None
    with pytest.raises(EvalError, match=r"no per_query\.csv"):
        load_system(tmp_path / "missing")
    broken = tmp_path / "broken.csv"
    pd.DataFrame({"query_id": ["a"], "rr": [1.0]}).to_csv(broken, index=False)
    with pytest.raises(EvalError, match="missing columns"):
        load_system(broken)


def test_full_comparison_orders_systems_and_writes_every_part():
    rng = np.random.default_rng(2)
    n = 240
    groups = np.repeat(np.arange(80), 3).astype(str)
    participants = np.tile(["P1", "P2", "P3", "P4"], n // 4)
    weak = _table(rng.uniform(0.0, 0.2, n))
    strong = _table(np.clip(weak["rr"].to_numpy() + 0.3, 0, 1))
    mid = _table(np.clip(weak["rr"].to_numpy() + 0.15, 0, 1))
    summary, pairs, tex = compare(
        {"weak": weak, "mid": mid, "strong": strong},
        groups,
        participants,
        n_library=100,
        n_boot=500,
        n_perm=2000,
        seed=0,
    )
    mrr = summary[summary["metric"] == "MRR"].set_index("system")["mean"]
    assert mrr["strong"] > mrr["mid"] > mrr["weak"]
    assert len(pairs) == 3
    assert {"p_value_holm", "participants_a_better"} <= set(pairs.columns)
    strong_vs_weak = pairs[(pairs.system_a == "weak") & (pairs.system_b == "strong")].iloc[0]
    assert strong_vs_weak["mean_difference"] < 0
    assert strong_vs_weak["participants_b_better"] == 4
    assert r"\begin{tabular}" in tex
    assert "Random chance" in tex
    assert tex.count("\\\\") >= 4


def test_latex_escapes_special_characters_and_handles_missing_chance():
    summary = {
        "onset_3band & more": {
            "MRR": (0.3, 0.2, 0.4),
            "Hit@1": (0.2, 0, 0),
            "Hit@5": (0.4, 0, 0),
            "Hit@10": (0.5, 0, 0),
            "mAP": (0.25, 0, 0),
        }
    }
    tex = latex_table(summary, {"MRR": None})
    assert r"onset\_3band \& more" in tex
    assert "Random chance" not in tex
    assert "0.300 (0.200--0.400)" in tex
