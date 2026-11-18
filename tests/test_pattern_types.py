"""tools.pattern_types: loop types, matching queries to them, MRR per type, the chart files."""

from __future__ import annotations

import argparse
import dataclasses

import pandas as pd
import pytest

from src.eval.evaluate import EvalError
from src.music.spec import PatternTypes, load_spec
from tools import pattern_types as pt

# Small thresholds so a few hits make each density; the real ones are tested in test_spec.
SPEC = dataclasses.replace(
    load_spec(),
    pattern_types=PatternTypes(0.2, 0.7, 0.5, sparse_below_sounds=3, busy_from_sounds=6),
)
SPB = SPEC.same_rhythm.steps_per_beat
BEATS = 8


def record(clip_id: str, hits: dict[str, list[int]]) -> dict:
    steps = BEATS * SPB
    return {
        "clip_id": clip_id,
        "pattern_id": f"p_{clip_id}",
        "split": "test",
        "tempo_bpm": 100.0,
        "source_performance": clip_id.split(":")[0],
        "start_bar": 0,
        "start_beat": 0.0,
        "end_beat": float(BEATS),
        "grid": {
            "voices": list(SPEC.voices),
            "steps_per_beat": SPB,
            "onsets": [[s in hits.get(v, []) for s in range(steps)] for v in SPEC.voices],
        },
    }


SIXTEENTH, TRIPLET = SPB // 4, SPB // 3
RECORDS = [
    # straight and sparse: one 8th-note off-beat
    record("d1/s1/1:bar0", {"kick": [0], "hats": [2 * SIXTEENTH]}),
    # swung and medium: off-beats on the last triplet of beats 1 and 2
    record("d2/s1/1:bar0", {"hats": [0, 2 * TRIPLET, SPB, SPB + 2 * TRIPLET]}),
    # no feel (half straight, half triplet) and busy: 7 sounds
    record("d3/s1/1:bar0", {"kick": [0, SPB, 2 * SPB, 3 * SPB, 4 * SPB], "hats": [3, 4]}),
]
MANIFEST = pd.DataFrame(
    {
        "query_id": ["q1", "q2", "q3", "q4"],
        "source_clip_id": ["d1/s1/1:bar0", "d1/s1/1:bar0", "d2/s1/1:bar0", "d3/s1/1:bar0"],
        "participant_id": ["P1", "P2", "P1", "P2"],
        "source": "synthetic",
    }
)
RR = {"q1": 1.0, "q2": 0.5, "q3": 0.25, "q4": 1 / 3}


def per_query(rr: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"query_id": q, "rr": r, "ap": r, "hit@1": float(r == 1), "hit@5": 1.0, "hit@10": 1.0}
        for q, r in rr.items()
    ]
    return pd.DataFrame(rows)


def test_loop_types():
    loops = pt.loop_types(RECORDS, SPEC).set_index("clip_id")
    assert loops["feel"].fillna("none").tolist() == ["straight", "swung", "none"]
    assert loops["density"].tolist() == ["sparse", "medium", "busy"]
    assert loops["sounds"].tolist() == [2, 4, 7]


def test_summarize_gives_mrr_and_counts_per_type():
    loops = pt.loop_types(RECORDS, SPEC)
    systems = {"A": per_query(RR).set_index("query_id")}
    queries = pt.query_types(systems["A"].index, MANIFEST, loops)
    s = pt.summarize(systems, queries, n_boot=200).set_index("type")
    assert s.loc["straight", "mrr"] == pytest.approx(0.75)
    assert s.loc["swung", "mrr"] == pytest.approx(0.25)
    assert s.loc["busy", "mrr"] == pytest.approx(1 / 3)
    assert s.loc["straight", ["queries", "loops", "people"]].tolist() == [2, 1, 2]
    assert (s["ci_low"] <= s["mrr"] + 1e-12).all() and (s["mrr"] <= s["ci_high"] + 1e-12).all()
    assert len(s) == 5  # the loop with no feel is only in its density column


def test_query_types_rejects_unknown_queries_and_outside_loops():
    loops = pt.loop_types(RECORDS, SPEC)
    with pytest.raises(EvalError, match="not in the query manifest"):
        pt.query_types(pd.Index(["q1", "q9"]), MANIFEST, loops)
    outside = MANIFEST.assign(source_clip_id=["d9/s1/1:bar0"] * 4)
    with pytest.raises(EvalError, match="outside the GMD test split"):
        pt.query_types(pd.Index(["q1"]), outside, loops)


def test_parse_system():
    assert pt.parse_system("R1 (onsets)=results/eval/r1") == (
        "R1 (onsets)",
        pt.Path("results/eval/r1"),
    )
    with pytest.raises(argparse.ArgumentTypeError):
        pt.parse_system("results/eval/r1")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "load_spec", lambda: SPEC)
    monkeypatch.setattr(pt, "load_catalog", lambda *a, **k: RECORDS)
    manifest = tmp_path / "query_manifest.csv"
    MANIFEST.to_csv(manifest, index=False)
    for name, scale in (("a", 1.0), ("b", 0.5)):
        (tmp_path / name).mkdir()
        per_query({q: r * scale for q, r in RR.items()}).to_csv(
            tmp_path / name / "per_query.csv", index=False
        )
    return tmp_path, manifest


def test_main_writes_the_chart_and_tables(setup):
    tmp, manifest = setup
    out = tmp / "out"
    args = ["--system", f"Onset={tmp / 'a'}", "--system", f"R1={tmp / 'b' / 'per_query.csv'}"]
    rc = pt.main([*args, "--query-manifest", str(manifest), "--out", str(out), "--n-boot", "200"])
    assert rc == 0
    assert (out / "a_pattern_types.pdf").read_bytes().startswith(b"%PDF")
    assert (out / "a_pattern_types.png").stat().st_size > 0
    table = pd.read_csv(out / "pattern_types.csv")
    assert len(table) == 2 * 5 and set(table["system"]) == {"Onset", "R1"}
    assert len(pd.read_csv(out / "loop_types.csv")) == len(RECORDS)


def test_main_rejects_systems_scored_on_different_queries(setup, capsys):
    tmp, manifest = setup
    per_query({"q1": 1.0}).to_csv(tmp / "b" / "per_query.csv", index=False)
    args = ["--system", f"Onset={tmp / 'a'}", "--system", f"R1={tmp / 'b'}"]
    rc = pt.main([*args, "--query-manifest", str(manifest), "--out", str(tmp / "out")])
    assert rc == 1
    assert "different queries" in capsys.readouterr().out
    assert not (tmp / "out").exists()


def test_main_needs_distinct_names(setup):
    tmp, manifest = setup
    with pytest.raises(SystemExit):
        pt.main(["--system", f"A={tmp / 'a'}", "--system", f"A={tmp / 'b'}"])
