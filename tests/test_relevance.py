"""Tests for src/eval/relevance.py, on small fake manifests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.eval.evaluate import EvalError, evaluate, load_relevance
from src.eval.relevance import (
    LIBRARY_COLUMNS,
    build_relevance,
    read_manifest,
    write_relevance,
)


def _library(rows):
    return pd.DataFrame(rows, columns=["clip_id", "pattern_id", "dualized_pattern_id", "split"])


def _queries(rows):
    return pd.DataFrame(
        rows,
        columns=["query_id", "pattern_id", "dualized_pattern_id", "query_type", "source", "split"],
    )


@pytest.fixture
def library():
    # Patterns A and B share the tap pattern T1; C has T2. A is on two test kits and also
    # appears in train. D exists only in train.
    return _library(
        [
            ("a_kit1", "A", "T1", "test"),
            ("a_kit2", "A", "T1", "test"),
            ("b_kit1", "B", "T1", "test"),
            ("c_kit1", "C", "T2", "test"),
            ("a_train", "A", "T1", "train"),
            ("d_train", "D", "T3", "train"),
        ]
    )


def test_beatbox_query_matches_same_pattern_on_every_test_clip(library):
    queries = _queries([("q1", "A", "T1", "beatbox", "synthetic", "test")])
    relevance, _ = build_relevance(library, queries)
    assert relevance == {"q1": ["a_kit1", "a_kit2"]}  # not b_kit1, not the train clip


def test_tap_query_matches_by_dualized_pattern(library):
    queries = _queries([("t1", "A", "T1", "tap", "synthetic", "test")])
    relevance, _ = build_relevance(library, queries)
    assert relevance == {"t1": ["a_kit1", "a_kit2", "b_kit1"]}


def test_tap_query_without_a_3_voice_pattern_still_scores(library):
    # TapTamDrum has no 3-voice pattern, only a dualized one.
    queries = _queries([("t2", None, "T2", "tap", "taptamdrum", "test")])
    relevance, report = build_relevance(library, queries)
    assert relevance == {"t2": ["c_kit1"]}
    assert report["by_testset"]["tap/taptamdrum"]["n_scoreable"] == 1


def test_unscoreable_queries_are_dropped_and_counted(library):
    queries = _queries(
        [
            ("nopat", None, "T1", "beatbox", "team", "test"),
            ("nomatch", "D", "T3", "beatbox", "team", "test"),  # D is only in train
            ("ok", "C", "T2", "beatbox", "team", "test"),
        ]
    )
    relevance, report = build_relevance(library, queries)
    assert list(relevance) == ["ok"]
    group = report["by_testset"]["beatbox/team"]
    assert group["dropped_no_pattern_id"] == 1
    assert group["dropped_no_clip_in_pool"] == 1
    assert group["n_queries"] == 3


def test_only_the_chosen_query_split_is_built(library):
    queries = _queries(
        [
            ("q_test", "A", "T1", "beatbox", "synthetic", "test"),
            ("q_train", "A", "T1", "beatbox", "synthetic", "train"),
        ]
    )
    relevance, report = build_relevance(library, queries)
    assert list(relevance) == ["q_test"]
    assert report["queries_in_other_splits_ignored"] == 1


def test_report_flags_patterns_shared_with_other_splits(library):
    queries = _queries([("q1", "A", "T1", "beatbox", "synthetic", "test")])
    _, report = build_relevance(library, queries)
    assert report["pool_patterns_shared_with_other_splits"] == 1  # pattern A
    assert report["pool_clips"] == 4
    assert report["by_testset"]["beatbox/synthetic"]["relevant_set_size"]["max"] == 2


def test_non_exact_match_rule_is_refused(library):
    queries = _queries([("q1", "A", "T1", "beatbox", "synthetic", "test")])
    with pytest.raises(EvalError, match="exact"):
        build_relevance(library, queries, match="within_2_hits")


def test_bad_manifests_are_rejected(library):
    good = _queries([("q1", "A", "T1", "beatbox", "synthetic", "test")])
    with pytest.raises(EvalError, match="duplicate clip_id"):
        build_relevance(pd.concat([library, library.iloc[:1]]), good)
    with pytest.raises(EvalError, match="duplicate query_id"):
        build_relevance(library, pd.concat([good, good]))
    with pytest.raises(EvalError, match="beatbox or tap"):
        build_relevance(library, _queries([("q", "A", "T1", "hum", "team", "test")]))
    with pytest.raises(EvalError, match="empty values in 'pattern_id'"):
        build_relevance(_library([("c", None, "T1", "test")]), good)
    with pytest.raises(EvalError, match="no clips"):
        build_relevance(library, good, pool_split="validation")


def test_manifest_reader_treats_only_empty_cells_as_null(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("clip_id,pattern_id,dualized_pattern_id,split\nc1,null,  ,test\n")
    frame = read_manifest(path, LIBRARY_COLUMNS)
    assert frame.loc[0, "pattern_id"] == "null"  # a real value, not a missing one
    assert pd.isna(frame.loc[0, "dualized_pattern_id"])  # blank after stripping
    with pytest.raises(EvalError, match="missing columns"):
        read_manifest(path, (*LIBRARY_COLUMNS, "kit_id"))


def test_file_round_trip_and_layout(library, tmp_path):
    queries = _queries(
        [
            ("q2", "C", "T2", "beatbox", "synthetic", "test"),
            ("q1", "A", "T1", "beatbox", "synthetic", "test"),
        ]
    )
    relevance, _ = build_relevance(library, queries)
    path = tmp_path / "relevance.json"
    write_relevance(relevance, path)
    assert json.loads(path.read_text()) == relevance
    assert path.read_text().splitlines()[1].startswith('"q1"')  # sorted, one query per line
    assert load_relevance(path) == relevance
    write_relevance({}, path)
    assert json.loads(path.read_text()) == {}


def test_key_works_with_the_scoring_script(library, tmp_path):
    # A system that ranks the right clips first must score MRR 1 against this key.
    queries = _queries(
        [
            ("q1", "A", "T1", "beatbox", "synthetic", "test"),
            ("q2", "C", "T2", "beatbox", "synthetic", "test"),
        ]
    )
    relevance, _ = build_relevance(library, queries)
    pool_ids = ["a_kit1", "a_kit2", "b_kit1", "c_kit1"]
    scores = np.array([[1.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.1, 1.0]])
    np.savez(
        tmp_path / "s.npz",
        query_ids=np.array(["q1", "q2"]),
        library_ids=np.array(pool_ids),
        scores=scores,
    )
    table, _ = evaluate(tmp_path / "s.npz", relevance)
    assert table["rr"].tolist() == [1.0, 1.0]
    assert table["n_relevant"].tolist() == [2, 1]


def test_pool_kits_limits_candidates_to_the_test_kits():
    # Test patterns on test kits only: a test-split clip on another kit is not a candidate.
    library = pd.DataFrame(
        [
            ("a_sal", "A", "T1", "test", "salamander"),
            ("a_mul", "A", "T1", "test", "muldjordkit"),
            ("a_avl", "A", "T1", "test", "avl_black_pearl"),
        ],
        columns=["clip_id", "pattern_id", "dualized_pattern_id", "split", "kit_id"],
    )
    queries = _queries([("q1", "A", "T1", "beatbox", "synthetic", "test")])
    relevance, report = build_relevance(library, queries, pool_kits=["salamander", "muldjordkit"])
    assert relevance == {"q1": ["a_mul", "a_sal"]}
    assert report["pool_kits"] == ["muldjordkit", "salamander"]
    assert report["pool_clips"] == 2
    everything, report_all = build_relevance(library, queries)
    assert len(everything["q1"]) == 3
    assert report_all["pool_kits"] == "all"


def test_pool_kits_errors_are_clear(library):
    queries = _queries([("q1", "A", "T1", "beatbox", "synthetic", "test")])
    with pytest.raises(EvalError, match="kit_id"):
        build_relevance(library, queries, pool_kits=["salamander"])  # fixture has no kit_id
    with_kits = library.assign(kit_id="salamander")
    with pytest.raises(EvalError, match="no 'test' clips on kits"):
        build_relevance(with_kits, queries, pool_kits=["muldjordkit"])
